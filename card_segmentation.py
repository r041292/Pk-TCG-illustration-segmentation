"""Segmenta, rectifica y recorta cartas a partir de imágenes preprocesadas.

La estrategia combina mapas Otsu en luminancia/saturación, bordes y contornos
geométricos. Un contorno activo inicializado sobre el mejor rectángulo refina
la evidencia de borde de manera conservadora; si se vuelve inestable, se usa
el rectángulo geométrico original.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import sobel
from skimage.measure import LineModelND, ransac
from skimage.segmentation import active_contour


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CARD_RATIO = 63 / 88  # ancho / alto de una carta Pokémon estándar


def resize_for_detection(image: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image, scale
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA), scale


def order_points(points: np.ndarray) -> np.ndarray:
    """Ordena puntos como superior-izq., superior-der., inf.-der., inf.-izq."""
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    difference = np.diff(points, axis=1).ravel()
    ordered[1] = points[np.argmin(difference)]
    ordered[3] = points[np.argmax(difference)]
    return ordered


def rectangle_metrics(box: np.ndarray, contour: np.ndarray, shape: tuple[int, int]) -> tuple[float, float, float] | None:
    width, height = cv2.minAreaRect(box.astype(np.float32))[1]
    if min(width, height) < 20:
        return None
    ratio = min(width, height) / max(width, height)
    area_fraction = (width * height) / (shape[0] * shape[1])
    rectangularity = cv2.contourArea(contour) / max(width * height, 1)
    return ratio, area_fraction, rectangularity


def candidate_score(box: np.ndarray, contour: np.ndarray, shape: tuple[int, int]) -> float | None:
    metrics = rectangle_metrics(box, contour, shape)
    if metrics is None:
        return None
    ratio, area_fraction, rectangularity = metrics
    # Una ilustración interna puede tener proporción de carta, pero ocupa una
    # fracción mucho menor que la carta completa. El mínimo evita esos falsos
    # positivos sin excluir cartas fotografiadas dentro de cápsulas.
    if not 0.47 <= ratio <= 0.88 or not 0.08 <= area_fraction <= 0.75:
        return None
    ordered = order_points(box)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    side_height = np.linalg.norm(ordered[3] - ordered[0])
    # Con orientación EXIF ya normalizada, la carta debe ser vertical. Esta
    # validación elimina la ilustración interna, que suele ser horizontal.
    if top_width >= side_height:
        return None
    center = box.mean(axis=0)
    image_center = np.array([shape[1] / 2, shape[0] / 2])
    center_penalty = np.linalg.norm(center - image_center) / np.linalg.norm(image_center)
    ratio_score = 1 - min(abs(ratio - CARD_RATIO) / 0.22, 1)
    # Se penalizan rectángulos enormes (frecuentemente el estuche completo).
    size_score = 1 - abs(area_fraction - 0.28) / 0.55
    return 3.0 * ratio_score + 1.2 * rectangularity + max(size_score, 0) - 0.5 * center_penalty


def adaptive_canny(channel: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Calcula umbrales Canny desde la mediana del canal de la imagen."""
    nonzero = channel[channel > 0]
    median = float(np.median(nonzero)) if nonzero.size else float(np.median(channel))
    lower = int(max(20, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    # Evita un intervalo degenerado en fondos casi negros o blancos.
    upper = max(upper, lower + 20)
    return cv2.Canny(channel, lower, upper, L2gradient=True)


def reflection_suppressed_luminance(luminance: np.ndarray, saturation: np.ndarray, value: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Atenúa reflejos con brillo alto, baja saturación y baja cromaticidad.

    Es una aproximación de polarización basada en color: los reflejos del
    plástico suelen ser casi blancos, mientras que bordes impresos mantienen
    diferencia entre canales. Solo se usa para obtener bordes, no para guardar
    el recorte visual.
    """
    channel_difference = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    bright = value >= np.percentile(value, 88)
    low_saturation = saturation <= max(35, np.percentile(saturation, 35))
    reflection_mask = (bright & low_saturation & (channel_difference <= 32)).astype(np.uint8) * 255
    reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.inpaint(luminance, reflection_mask, 3, cv2.INPAINT_TELEA)


def build_edge_map(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Crea máscaras Otsu y un mapa de bordes usando LAB y HSV."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    luminance = lab[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    clahe_l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(luminance)
    # La variante de supresión de reflejos se evaluó en la muestra inicial,
    # pero favoreció falsos rectángulos en cartas claras; se conserva la
    # luminancia base como ruta estable hasta disponer de una mejor compuerta.
    polarized_l = clahe_l

    _, otsu_l = cv2.threshold(clahe_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_s = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # A diferencia de Otsu, este umbral compara cada píxel con su vecindario:
    # mantiene los bordes de carta cuando una sombra cruza solo una zona.
    adaptive_mask = cv2.adaptiveThreshold(
        clahe_l, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 41, 6,
    )
    # Otsu sobre saturación separa gran parte de la carta coloreada del
    # plástico/fondo. Es una fuente adicional de candidatos, no una máscara
    # definitiva: algunas cartas son oscuras o poco saturadas.
    mask = cv2.morphologyEx(otsu_s, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    # Las cartas grises/blancas pueden perderse en HSV. La región más oscura
    # que el fondo, obtenida por Otsu en L, crea candidatos complementarios.
    luminance_mask = cv2.bitwise_not(otsu_l)
    luminance_mask = cv2.morphologyEx(luminance_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    luminance_mask = cv2.morphologyEx(luminance_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    # Tres fuentes de bordes: L detecta cambios de iluminación, V conserva
    # contraste bajo variaciones de color y el gradiente de S recupera bordes
    # cromáticos que el brillo del plástico puede ocultar.
    saturation_gradient = cv2.convertScaleAbs(cv2.Sobel(saturation, cv2.CV_16S, 1, 1, ksize=3))
    edges_l = adaptive_canny(clahe_l)
    edges_polarized = adaptive_canny(polarized_l)
    edges_s = adaptive_canny(saturation_gradient, sigma=0.20)
    edges = cv2.bitwise_or(edges_l, edges_polarized)
    edges = cv2.bitwise_or(edges, edges_s)
    # Reconecta un borde de carta interrumpido por reflejos, texto o arte
    # holográfico. Se mantiene como una fuente separada para no sustituir el
    # detalle fino de Canny ni forzar una máscara por color.
    structural_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    structural_edges = cv2.morphologyEx(structural_edges, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=1)
    return edges, mask, luminance_mask, structural_edges, adaptive_mask


def find_card_rectangle(image: np.ndarray, use_legacy_edges: bool = False) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, bool]:
    edges, mask, luminance_mask, structural_edges, adaptive_mask = build_edge_map(image)
    if use_legacy_edges:
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
        legacy_l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(luminance)
        edges = cv2.Canny(legacy_l, 45, 130)
        structural_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        structural_edges = cv2.morphologyEx(structural_edges, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=1)
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    color_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    luminance_contours, _ = cv2.findContours(luminance_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    structural_contours, _ = cv2.findContours(structural_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = edge_contours + color_contours + luminance_contours + structural_contours
    best_score, best_box, best_from_hough = -np.inf, None, False

    for contour in contours:
        if len(contour) < 4:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        score = candidate_score(box, contour, image.shape[:2])
        if score is not None and score > best_score:
            best_score, best_box, best_from_hough = score, box, False

    # La jerarquía separa rectángulos anidados: por ejemplo, la carta dentro
    # de una cápsula PSA. El candidato interior recibe un bono solo cuando su
    # contenedor es claramente mayor y también parece rectangular.
    tree_contours, hierarchy = cv2.findContours(adaptive_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        parents = hierarchy[0, :, 3]
        # Etiquetas PSA: rectángulos horizontales relativamente pequeños en
        # la parte superior. Sirven como ancla espacial, no como detección.
        labels: list[np.ndarray] = []
        for contour in tree_contours:
            if len(contour) < 4:
                continue
            label_box = order_points(cv2.boxPoints(cv2.minAreaRect(contour)))
            label_width = np.linalg.norm(label_box[1] - label_box[0])
            label_height = np.linalg.norm(label_box[3] - label_box[0])
            label_area = abs(cv2.contourArea(label_box.astype(np.float32))) / (image.shape[0] * image.shape[1])
            if label_width >= 1.45 * label_height and 0.008 <= label_area <= 0.18 and label_box[:, 1].mean() < image.shape[0] * 0.45:
                labels.append(label_box)

        for index, contour in enumerate(tree_contours):
            parent_index = parents[index]
            if parent_index < 0 or len(contour) < 4 or len(tree_contours[parent_index]) < 4:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            parent_box = cv2.boxPoints(cv2.minAreaRect(tree_contours[parent_index]))
            score = candidate_score(box, contour, image.shape[:2])
            parent_metrics = rectangle_metrics(parent_box, tree_contours[parent_index], image.shape[:2])
            if score is None or parent_metrics is None:
                continue
            child_area = abs(cv2.contourArea(box.astype(np.float32)))
            parent_area = abs(cv2.contourArea(parent_box.astype(np.float32)))
            area_ratio = child_area / max(parent_area, 1)
            center_inside = cv2.pointPolygonTest(parent_box.astype(np.float32), tuple(box.mean(axis=0)), False) >= 0
            if 0.10 <= area_ratio <= 0.82 and center_inside:
                nested_score = score + 0.65
                if nested_score > best_score:
                    best_score, best_box, best_from_hough = nested_score, box, False

            # Regla PSA: la carta es un rectángulo vertical inmediatamente
            # debajo de la etiqueta, no el rectángulo que contiene ambos.
            candidate_box = order_points(box)
            candidate_top = candidate_box[:, 1].min()
            candidate_center_x = candidate_box[:, 0].mean()
            for label_box in labels:
                label_bottom = label_box[:, 1].max()
                label_center_x = label_box[:, 0].mean()
                vertical_gap = candidate_top - label_bottom
                if -0.02 * image.shape[0] <= vertical_gap <= 0.30 * image.shape[0] and abs(candidate_center_x - label_center_x) <= 0.25 * image.shape[1]:
                    psa_score = score + 1.05
                    if psa_score > best_score:
                        best_score, best_box, best_from_hough = psa_score, box, False

        # Los contornos de Canny/Hough no tienen necesariamente una relación
        # padre-hijo en la máscara local. También se evalúan contra la etiqueta
        # para no perder el borde exterior de una carta tenue.
        for contour in contours:
            if len(contour) < 4:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            score = candidate_score(box, contour, image.shape[:2])
            if score is None:
                continue
            candidate_box = order_points(box)
            candidate_top = candidate_box[:, 1].min()
            candidate_center_x = candidate_box[:, 0].mean()
            for label_box in labels:
                vertical_gap = candidate_top - label_box[:, 1].max()
                if -0.02 * image.shape[0] <= vertical_gap <= 0.30 * image.shape[0] and abs(candidate_center_x - label_box[:, 0].mean()) <= 0.25 * image.shape[1]:
                    psa_score = score + 1.05
                    if psa_score > best_score:
                        best_score, best_box, best_from_hough = psa_score, box, False

    # Cuando Canny deja el perímetro fragmentado, Hough conserva las aristas
    # largas. Se ensamblan pares de líneas horizontal/vertical en rectángulos
    # con la proporción física esperada de una carta.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 55, minLineLength=max(45, min(image.shape[:2]) // 7), maxLineGap=28)
    if lines is not None:
        horizontal, vertical = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length and abs(dy) <= 0.16 * length:
                horizontal.append(((y1 + y2) / 2, length))
            elif length and abs(dx) <= 0.16 * length:
                vertical.append(((x1 + x2) / 2, length))

        def distinct(lines_data: list[tuple[float, float]]) -> list[float]:
            selected: list[tuple[float, float]] = []
            for line in sorted(lines_data, key=lambda item: item[1], reverse=True):
                if all(abs(line[0] - chosen[0]) > 12 for chosen in selected):
                    selected.append(line)
            return [line[0] for line in selected[:12]]

        horizontal, vertical = distinct(horizontal), distinct(vertical)
        strength = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (5, 5), 0)
        for top in horizontal:
            for bottom in horizontal:
                for left in vertical:
                    for right in vertical:
                        if bottom - top < 60 or right - left < 45:
                            continue
                        box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
                        base = candidate_score(box, box.reshape(-1, 1, 2), image.shape[:2])
                        if base is None:
                            continue
                        t, b, l, r = map(int, (top, bottom, left, right))
                        perimeter = np.concatenate((strength[t - 2:t + 3, l:r].ravel(), strength[b - 2:b + 3, l:r].ravel(), strength[t:b, l - 2:l + 3].ravel(), strength[t:b, r - 2:r + 3].ravel()))
                        # La geometría debe prevalecer sobre el brillo del
                        # estuche, que puede crear líneas Hough muy fuertes.
                        score = base + 0.5 * float(perimeter.mean())
                        if score > best_score:
                            best_score, best_box, best_from_hough = score, box, True

    # El Canny fijo conserva una ruta de compatibilidad para fotografías donde
    # el umbral adaptativo sea demasiado selectivo.
    if best_box is None and not use_legacy_edges:
        return find_card_rectangle(image, use_legacy_edges=True)
    return best_box, edges, cv2.bitwise_or(mask, adaptive_mask), best_from_hough


def refine_with_active_contour(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Ajusta el borde local, pero descarta el resultado si se aleja demasiado."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    edge_energy = sobel(gray)
    box = order_points(box)
    samples_per_side = 30
    snake_xy = []
    for start, end in zip(box, np.roll(box, -1, axis=0)):
        snake_xy.extend(np.linspace(start, end, samples_per_side, endpoint=False))
    snake_xy = np.asarray(snake_xy)
    snake_rc = snake_xy[:, ::-1]

    try:
        result_rc = active_contour(
            edge_energy, snake_rc, alpha=0.015, beta=4.0, gamma=0.001,
            w_line=0, w_edge=2.0, max_num_iter=120, convergence=0.1,
        )
        refined = cv2.boxPoints(cv2.minAreaRect(result_rc[:, ::-1].astype(np.float32)))
        original_area = abs(cv2.contourArea(box.astype(np.float32)))
        refined_area = abs(cv2.contourArea(refined.astype(np.float32)))
        displacement = np.mean(np.linalg.norm(order_points(refined) - box, axis=1))
        diagonal = np.linalg.norm(box[0] - box[2])
        if 0.72 <= refined_area / max(original_area, 1) <= 1.28 and displacement <= 0.10 * diagonal:
            return refined
    except (ValueError, np.linalg.LinAlgError):
        pass
    return box


def line_intersection(first: LineModelND, second: LineModelND) -> np.ndarray | None:
    """Calcula la intersección de dos rectas de skimage en coordenadas x, y."""
    # scikit-image expone estos valores como atributos separados en versiones
    # recientes, pero en versiones anteriores (como algunas de Colab) solo
    # están disponibles en ``params``.
    if hasattr(first, "origin"):
        origin_a, direction_a = first.origin, first.direction
    else:
        origin_a, direction_a = first.params
    if hasattr(second, "origin"):
        origin_b, direction_b = second.origin, second.direction
    else:
        origin_b, direction_b = second.params
    normal_a = np.array([-direction_a[1], direction_a[0]])
    normal_b = np.array([-direction_b[1], direction_b[0]])
    system = np.vstack((normal_a, normal_b))
    if abs(np.linalg.det(system)) < 1e-5:
        return None
    return np.linalg.solve(system, np.array((normal_a @ origin_a, normal_b @ origin_b)))


def refine_with_ransac(edges: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Refina un rectángulo con cuatro líneas robustas ajustadas por RANSAC.

    Los puntos de cada lado se restringen a una banda alrededor del rectángulo
    inicial. Esto permite descartar arte, texto y reflejos sin buscar líneas
    irrelevantes en toda la fotografía.
    """
    box = order_points(box)
    edge_y, edge_x = np.nonzero(edges)
    points = np.column_stack((edge_x, edge_y)).astype(float)
    if len(points) < 80:
        return box

    side_models: list[LineModelND] = []
    short_side = min(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[3] - box[0]))
    band = max(5.0, 0.05 * short_side)
    for start, end in zip(box, np.roll(box, -1, axis=0)):
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1:
            return box
        unit = direction / length
        relative = points - start
        projection = relative @ unit
        distance = np.abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
        side_points = points[(projection >= -0.12 * length) & (projection <= 1.12 * length) & (distance <= band)]
        if len(side_points) < 20:
            return box
        try:
            model, inliers = ransac(
                side_points, LineModelND, min_samples=2, residual_threshold=2.5,
                max_trials=150, rng=42,
            )
        except (ValueError, np.linalg.LinAlgError):
            return box
        if model is None or inliers.sum() < 16:
            return box
        side_models.append(model)

    corners = []
    for previous, current in zip([side_models[-1], *side_models[:-1]], side_models):
        corner = line_intersection(previous, current)
        if corner is None:
            return box
        corners.append(corner)
    refined = order_points(np.asarray(corners, dtype=np.float32))

    original_area = abs(cv2.contourArea(box.astype(np.float32)))
    refined_area = abs(cv2.contourArea(refined.astype(np.float32)))
    movement = np.mean(np.linalg.norm(refined - box, axis=1))
    diagonal = np.linalg.norm(box[0] - box[2])
    refined_ratio = min(np.linalg.norm(refined[1] - refined[0]), np.linalg.norm(refined[3] - refined[0]))
    refined_ratio /= max(np.linalg.norm(refined[3] - refined[0]), np.linalg.norm(refined[1] - refined[0]), 1)
    if 0.72 <= refined_area / max(original_area, 1) <= 1.32 and movement <= 0.12 * diagonal and abs(refined_ratio - CARD_RATIO) <= 0.15:
        return refined
    return box


def warp_card(image: np.ndarray, box: np.ndarray, output_width: int = 1000) -> np.ndarray:
    source = order_points(box)
    target_height = round(output_width / CARD_RATIO)
    destination = np.array(
        [[0, 0], [output_width - 1, 0], [output_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (output_width, target_height), flags=cv2.INTER_CUBIC)


def diagnostic_image(image: np.ndarray, edges: np.ndarray, mask: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    overlay = image.copy()
    if box is not None:
        cv2.polylines(overlay, [order_points(box).astype(np.int32)], True, (0, 255, 0), 4)
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    height = 420
    panels = [overlay, edge_bgr, mask_bgr]
    resized = [cv2.resize(panel, (round(panel.shape[1] * height / panel.shape[0]), height)) for panel in panels]
    return cv2.hconcat(resized)


def process_image(source: Path, output_dir: Path) -> bool:
    image = cv2.imread(str(source))
    if image is None:
        print(f"No se pudo leer: {source.name}")
        return False
    detection_image, scale = resize_for_detection(image)
    # Si la foto ya tiene proporción de carta y el borde es coloreado, la
    # segmentación interna sería contraproducente: se conserva el encuadre.
    height, width = detection_image.shape[:2]
    aspect = min(width, height) / max(width, height)
    hsv = cv2.cvtColor(detection_image, cv2.COLOR_BGR2HSV)
    border = np.concatenate((hsv[:8, :, 1].ravel(), hsv[-8:, :, 1].ravel(), hsv[:, :8, 1].ravel(), hsv[:, -8:, 1].ravel()))
    already_cropped = abs(aspect - CARD_RATIO) < 0.055 and np.median(border) > 45
    if already_cropped:
        from_hough = False
        box = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        edges, mask, _, _, adaptive_mask = build_edge_map(detection_image)
        mask = cv2.bitwise_or(mask, adaptive_mask)
    else:
        box, edges, mask, from_hough = find_card_rectangle(detection_image)
    if box is not None:
        if from_hough:
            # Hough suele ubicar el borde interior bajo reflejos. Expandir
            # moderadamente evita perder texto o los bordes de la carta.
            center = box.mean(axis=0)
            box = center + 1.25 * (box - center)
            box[:, 0] = np.clip(box[:, 0], 0, width - 1)
            box[:, 1] = np.clip(box[:, 1], 0, height - 1)
        if not already_cropped:
            box = refine_with_ransac(edges, box)
            if not from_hough:
                box = refine_with_active_contour(detection_image, box)
        full_box = box / scale
        card = warp_card(image, full_box)
        cv2.imwrite(str(output_dir / f"{source.stem}_card.png"), card)
        status = "encuadre conservado" if already_cropped else "recortada"
    else:
        status = "sin candidato"
    diagnostic = diagnostic_image(detection_image, edges, mask, box)
    cv2.imwrite(str(output_dir / f"{source.stem}_debug.png"), diagnostic)
    print(f"{source.name}: {status}")
    return box is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmenta y recorta cartas Pokémon.")
    parser.add_argument("--input", type=Path, default=Path("img_pre"))
    parser.add_argument("--output", type=Path, default=Path("img_segm"))
    parser.add_argument("--limit", type=int, default=10, help="Máximo de imágenes; 0 procesa todas.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = sorted(path for path in args.input.iterdir() if path.suffix.lower() in VALID_EXTENSIONS)
    sources = sources[:args.limit] if args.limit else sources
    successes = sum(process_image(source, args.output) for source in sources)
    print(f"\nCompletado: {successes}/{len(sources)} cartas recortadas en {args.output}")


if __name__ == "__main__":
    main()
