"""Refina esquinas de recortes YOLO-OBB y rectifica las cartas por homografía."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from card_segmentation import CARD_RATIO, build_edge_map, order_points, refine_with_ransac, warp_card


# Ruta activa por defecto: compara la semilla YOLO (azul) contra RANSAC
# (verde). Para volver al comportamiento original, cambia este flag a False
# o ejecuta el script con --legacy-refinement.
USE_CANDIDATE_SELECTION = True
CANDIDATE_SCORE_MARGIN = 0.06
# Un candidato mixto puede recuperar bordes externos válidos aunque no gane
# con el margen más estricto usado para evitar refinamientos completos
# dudosos. Mantenerlo alineado con CANDIDATE_SCORE_MARGIN evita que una caja
# mixta coherente sea descartada por una diferencia arbitraria.
MIXED_SCORE_MARGIN = CANDIDATE_SCORE_MARGIN
CANDIDATE_SEARCH_RADIUS = 8
CANDIDATE_SIDE_START = 0.10
CANDIDATE_SIDE_END = 0.90
MIXED_AREA_RATIO_RANGE = (0.72, 1.32)
MIXED_MAX_MOVEMENT_RATIO = 0.12
MIXED_INTERNAL_PENALTY = 0.30
MIXED_MOVEMENT_PENALTY_SCALE = 0.50
MIXED_EXTERNAL_SAMPLE_OFFSET = 6.0
MIXED_INTERNAL_FRACTION = 0.70
BLUE_STABLE_CONFIDENCE = 0.95
BLUE_STABLE_GEOMETRY = 0.90
BLUE_STABLE_GOOD_SIDES = 3
# Guardia adicional para el borde externo: es ligeramente más permisiva que
# BLUE_STABLE_*. Se activa cuando el verde contiene al menos un lado interno
# respecto del azul, aunque los demás lados verdes expandan el área total.
# Así un borde interno aislado no puede ganar únicamente por textura local.
EXTERNAL_GUARD_CONFIDENCE = 0.95
EXTERNAL_GUARD_GEOMETRY = 0.82
EXTERNAL_GUARD_GOOD_SIDES = 2


def load_seed(metadata_path: Path, crop_shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    points = np.asarray(metadata["obb_points_crop_xy"], dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"{metadata_path.name}: se esperaban 4 puntos OBB.")
    height, width = crop_shape[:2]
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return order_points(points), metadata


def refine_corners(crop: np.ndarray, initial_box: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Ajusta líneas de borde alrededor de la semilla OBB con RANSAC.

    El refinamiento se acepta solo con las validaciones geométricas ya usadas
    por la etapa clásica. Si la evidencia de borde es ambigua, se conserva la
    detección de YOLO, que es una alternativa segura.
    """
    edges, _, _, _, _ = build_edge_map(crop)
    refined_box = order_points(refine_with_ransac(edges, initial_box))
    displacement = float(np.mean(np.linalg.norm(refined_box - initial_box, axis=1)))
    return refined_box, edges, displacement >= 0.75


def _longest_true_run(values: np.ndarray) -> float:
    """Devuelve la fracción de la secuencia cubierta por su tramo más largo."""
    longest = current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / max(len(values), 1)


def _luminance_and_strength(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene luminancia normalizada y magnitud de gradiente normalizada."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0
    smoothed = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 1.1)
    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    strength = cv2.magnitude(gradient_x, gradient_y)
    scale = max(float(np.percentile(strength, 95)), 1.0)
    strength = np.clip(strength / scale, 0.0, 1.0)
    return luminance, strength


def _sample_side_evidence(
    edges: np.ndarray,
    luminance: np.ndarray,
    strength: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> dict[str, float]:
    """Mide cobertura, continuidad, fuerza y contraste de un lado candidato."""
    height, width = edges.shape[:2]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1.0:
        return {"coverage": 0.0, "continuity": 0.0, "strength": 0.0, "contrast": 0.0, "score": 0.0}

    unit = direction / length
    normal = np.array([-unit[1], unit[0]], dtype=np.float32)
    sample_count = int(np.clip(round(length / 7.0), 80, 160))
    parameters = np.linspace(CANDIDATE_SIDE_START, CANDIDATE_SIDE_END, sample_count)
    points = start[None, :] + parameters[:, None] * direction[None, :]
    offsets = np.arange(-CANDIDATE_SEARCH_RADIUS, CANDIDATE_SEARCH_RADIUS + 1, dtype=np.float32)

    coverage_values: list[bool] = []
    closeness_values: list[float] = []
    strength_values: list[float] = []
    contrast_values: list[float] = []
    for point in points:
        best_distance = None
        best_strength = 0.0
        for offset in offsets:
            location = np.rint(point + offset * normal).astype(int)
            x, y = int(location[0]), int(location[1])
            if not (0 <= x < width and 0 <= y < height):
                continue
            if edges[y, x] > 0:
                distance = abs(float(offset))
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                weight = 1.0 - distance / (CANDIDATE_SEARCH_RADIUS + 1.0)
                best_strength = max(best_strength, float(strength[y, x]) * weight)

        covered = best_distance is not None
        coverage_values.append(covered)
        closeness_values.append(
            0.0 if best_distance is None else 1.0 - best_distance / (CANDIDATE_SEARCH_RADIUS + 1.0)
        )
        strength_values.append(best_strength)

        inner = np.rint(point - 4.0 * normal).astype(int)
        outer = np.rint(point + 4.0 * normal).astype(int)
        if (
            0 <= inner[0] < width
            and 0 <= inner[1] < height
            and 0 <= outer[0] < width
            and 0 <= outer[1] < height
        ):
            contrast_values.append(float(abs(luminance[inner[1], inner[0]] - luminance[outer[1], outer[0]])))
        else:
            contrast_values.append(0.0)

    coverage = float(np.mean(coverage_values))
    closeness = float(np.mean(closeness_values))
    continuity = _longest_true_run(np.asarray(coverage_values, dtype=bool))
    edge_strength = float(np.mean(strength_values))
    contrast = float(np.clip(np.mean(contrast_values) * 3.0, 0.0, 1.0))
    score = 0.50 * closeness + 0.25 * continuity + 0.15 * edge_strength + 0.10 * contrast
    return {
        "coverage": coverage,
        "continuity": continuity,
        "strength": edge_strength,
        "contrast": contrast,
        "score": float(score),
    }


def _geometry_score(box: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Puntúa proporción, simetría y permanencia dentro del recorte."""
    ordered = order_points(box)
    height, width = image_shape[:2]
    top_width = float(np.linalg.norm(ordered[1] - ordered[0]))
    bottom_width = float(np.linalg.norm(ordered[2] - ordered[3]))
    left_height = float(np.linalg.norm(ordered[3] - ordered[0]))
    right_height = float(np.linalg.norm(ordered[2] - ordered[1]))
    average_width = (top_width + bottom_width) / 2.0
    average_height = (left_height + right_height) / 2.0
    if min(average_width, average_height) < 1.0:
        return 0.0

    aspect = average_width / average_height
    aspect_score = max(0.0, 1.0 - abs(aspect - CARD_RATIO) / 0.20)
    width_symmetry = 1.0 - min(abs(top_width - bottom_width) / max(average_width, 1.0), 1.0)
    height_symmetry = 1.0 - min(abs(left_height - right_height) / max(average_height, 1.0), 1.0)
    symmetry_score = (width_symmetry + height_symmetry) / 2.0
    in_bounds = float(
        np.all(ordered[:, 0] >= 0)
        and np.all(ordered[:, 0] < width)
        and np.all(ordered[:, 1] >= 0)
        and np.all(ordered[:, 1] < height)
    )
    return float(0.70 * aspect_score + 0.20 * symmetry_score + 0.10 * in_bounds)


def _score_candidate(
    crop: np.ndarray,
    edges: np.ndarray,
    box: np.ndarray,
    luminance: np.ndarray | None = None,
    strength: np.ndarray | None = None,
) -> dict:
    """Calcula el puntaje auditable de un cuadrilátero candidato."""
    if luminance is None or strength is None:
        luminance, strength = _luminance_and_strength(crop)
    ordered = order_points(box)
    sides = [
        _sample_side_evidence(edges, luminance, strength, start, end)
        for start, end in zip(ordered, np.roll(ordered, -1, axis=0))
    ]
    edge_score = float(np.mean([side["score"] for side in sides]))
    geometry_score = _geometry_score(ordered, crop.shape[:2])
    total_score = 0.90 * edge_score + 0.10 * geometry_score
    return {
        "score": total_score,
        "edge_score": edge_score,
        "geometry_score": geometry_score,
        "sides": sides,
    }


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _line_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> np.ndarray | None:
    """Calcula la intersección de dos lados representados por sus extremos."""
    first_start = np.asarray(first_start, dtype=float)
    second_start = np.asarray(second_start, dtype=float)
    first_direction = np.asarray(first_end, dtype=float) - first_start
    second_direction = np.asarray(second_end, dtype=float) - second_start
    denominator = _cross(first_direction, second_direction)
    if abs(denominator) < 1e-6:
        return None
    parameter = _cross(second_start - first_start, second_direction) / denominator
    return first_start + parameter * first_direction


def _mixed_box(
    blue_box: np.ndarray, green_box: np.ndarray, side_choices: tuple[int, int, int, int]
) -> np.ndarray | None:
    """Construye un cuadrilátero usando para cada lado azul (0) o verde (1)."""
    blue = order_points(blue_box)
    green = order_points(green_box)
    sides = []
    for index, choice in enumerate(side_choices):
        source = green if choice else blue
        sides.append((source[index], source[(index + 1) % 4]))

    corners: list[np.ndarray] = []
    for index in range(4):
        previous_side = sides[(index - 1) % 4]
        current_side = sides[index]
        corner = _line_intersection(*previous_side, *current_side)
        if corner is None or not np.all(np.isfinite(corner)):
            return None
        corners.append(corner)
    return order_points(np.asarray(corners, dtype=np.float32))


def _is_valid_mixed_box(box: np.ndarray, blue_box: np.ndarray, image_shape: tuple[int, int]) -> bool:
    """Descarta mezclas cruzadas, degeneradas o incompatibles con la carta."""
    candidate = order_points(box)
    blue = order_points(blue_box)
    if not np.all(np.isfinite(candidate)):
        return False
    if not cv2.isContourConvex(candidate.astype(np.float32)):
        return False

    candidate_area = abs(float(cv2.contourArea(candidate.astype(np.float32))))
    blue_area = abs(float(cv2.contourArea(blue.astype(np.float32))))
    if candidate_area < 1.0 or not MIXED_AREA_RATIO_RANGE[0] <= candidate_area / max(blue_area, 1.0) <= MIXED_AREA_RATIO_RANGE[1]:
        return False

    blue_diagonal = max(float(np.linalg.norm(blue[0] - blue[2])), 1.0)
    movement = float(np.mean(np.linalg.norm(candidate - blue, axis=1)))
    if movement > MIXED_MAX_MOVEMENT_RATIO * blue_diagonal:
        return False

    height, width = image_shape[:2]
    allowance = 0.05 * max(width, height)
    if (
        np.any(candidate[:, 0] < -allowance)
        or np.any(candidate[:, 0] > width - 1 + allowance)
        or np.any(candidate[:, 1] < -allowance)
        or np.any(candidate[:, 1] > height - 1 + allowance)
    ):
        return False

    side_lengths = [float(np.linalg.norm(candidate[(index + 1) % 4] - candidate[index])) for index in range(4)]
    return min(side_lengths) >= 0.45 * min(
        float(np.linalg.norm(blue[1] - blue[0])), float(np.linalg.norm(blue[3] - blue[0]))
    )


def _green_sides_inside_blue(green_box: np.ndarray, blue_box: np.ndarray) -> list[bool]:
    """Marca lados verdes cuyo exterior inmediato aún está dentro del azul."""
    green = order_points(green_box)
    blue = order_points(blue_box)
    center = green.mean(axis=0)
    flags: list[bool] = []
    for start, end in zip(green, np.roll(green, -1, axis=0)):
        direction = end - start
        length = max(float(np.linalg.norm(direction)), 1.0)
        unit = direction / length
        normal = np.array([-unit[1], unit[0]], dtype=np.float32)
        midpoint = (start + end) / 2.0
        if float(np.dot(center - midpoint, normal)) < 0:
            normal = -normal
        outward = -normal
        inside_count = 0
        total_count = 0
        for parameter in np.linspace(CANDIDATE_SIDE_START, CANDIDATE_SIDE_END, 80):
            point = start + parameter * direction + MIXED_EXTERNAL_SAMPLE_OFFSET * outward
            inside_count += int(cv2.pointPolygonTest(blue.astype(np.float32), tuple(point), False) >= 0)
            total_count += 1
        flags.append(inside_count / max(total_count, 1) >= MIXED_INTERNAL_FRACTION)
    return flags


def _green_sides_allowed_by_external_guard(
    green_inside_flags: list[bool],
    blue_side_scores: np.ndarray,
    green_side_scores: np.ndarray,
    external_guard_active: bool,
) -> list[bool]:
    """Indica qué lados verdes aún pueden competir bajo la guardia externa.

    Un lado verde contenido dentro del azul representa un borde interno, por
    lo que se bloquea cuando el candidato azul tiene evidencia global fiable.
    Un lado verde que no está contenido sigue siendo elegible; la selección de
    las 16 combinaciones continúa decidiendo si realmente mejora el recorte.
    """
    if not external_guard_active:
        return [True] * 4

    allowed: list[bool] = []
    for index, is_inside in enumerate(green_inside_flags):
        if is_inside:
            allowed.append(False)
            continue
        # En lados no internos exigimos una ventaja clara para conservar el
        # beneficio de la comparación local sin favorecer diferencias mínimas.
        allowed.append(
            bool(
                green_side_scores[index] - blue_side_scores[index] >= 0.12
                and green_side_scores[index] >= 0.70
            )
        )
    return allowed


def _side_movement_penalties(blue_box: np.ndarray, green_box: np.ndarray) -> list[float]:
    """Calcula la penalización por desplazar cada lado desde YOLO."""
    blue = order_points(blue_box)
    green = order_points(green_box)
    penalties: list[float] = []
    for blue_start, blue_end, green_start, green_end in zip(
        blue,
        np.roll(blue, -1, axis=0),
        green,
        np.roll(green, -1, axis=0),
    ):
        side_length = max(float(np.linalg.norm(blue_end - blue_start)), 1.0)
        endpoint_movement = (
            float(np.linalg.norm(green_start - blue_start))
            + float(np.linalg.norm(green_end - blue_end))
        ) / 2.0
        movement_ratio = endpoint_movement / side_length
        penalties.append(0.18 * min(movement_ratio / 0.04, 1.0))
    return penalties


def select_preferred_candidate(
    crop: np.ndarray,
    edges: np.ndarray,
    initial_box: np.ndarray,
    refined_box: np.ndarray,
    yolo_confidence: float = 0.0,
) -> tuple[np.ndarray, bool, dict]:
    """Elige entre 16 combinaciones de lados y conserva azul si no hay mejora clara."""
    luminance, strength = _luminance_and_strength(crop)
    blue = _score_candidate(crop, edges, initial_box, luminance, strength)
    green = _score_candidate(crop, edges, refined_box, luminance, strength)
    blue_side_scores = np.asarray([side["score"] for side in blue["sides"]], dtype=float)
    green_side_scores = np.asarray([side["score"] for side in green["sides"]], dtype=float)
    side_movement_penalties = _side_movement_penalties(initial_box, refined_box)

    blue_stable = bool(
        yolo_confidence >= BLUE_STABLE_CONFIDENCE
        and blue["geometry_score"] >= BLUE_STABLE_GEOMETRY
        and int(np.sum(blue_side_scores >= 0.55)) >= BLUE_STABLE_GOOD_SIDES
    )
    green_area = abs(float(cv2.contourArea(order_points(refined_box).astype(np.float32))))
    blue_area = max(abs(float(cv2.contourArea(order_points(initial_box).astype(np.float32)))), 1.0)
    green_inside_flags = _green_sides_inside_blue(refined_box, initial_box)
    nested_green = bool(green_area / blue_area < 0.95 and sum(green_inside_flags) >= 3)
    green_internal_side_conflict = bool(any(green_inside_flags))
    blue_external_guard = bool(
        yolo_confidence >= EXTERNAL_GUARD_CONFIDENCE
        and blue["geometry_score"] >= EXTERNAL_GUARD_GEOMETRY
        and int(np.sum(blue_side_scores >= 0.55)) >= EXTERNAL_GUARD_GOOD_SIDES
    )
    external_guard_active = bool(blue_external_guard and green_internal_side_conflict)
    green_sides_allowed = _green_sides_allowed_by_external_guard(
        green_inside_flags,
        blue_side_scores,
        green_side_scores,
        external_guard_active,
    )
    internal_side_flags = (
        [not allowed for allowed in green_sides_allowed]
        if external_guard_active
        else [False] * 4
    )

    candidate_rows: list[dict] = []
    best_candidate: dict | None = None
    for side_choices in itertools.product((0, 1), repeat=4):
        candidate_box = _mixed_box(initial_box, refined_box, side_choices)
        if candidate_box is None or not _is_valid_mixed_box(candidate_box, initial_box, crop.shape[:2]):
            continue
        candidate_score = _score_candidate(crop, edges, candidate_box, luminance, strength)
        selected_side_scores = np.asarray(
            [green_side_scores[index] if side_choices[index] else blue_side_scores[index] for index in range(4)],
            dtype=float,
        )
        movement_penalty = float(
            MIXED_MOVEMENT_PENALTY_SCALE
            * np.mean([side_movement_penalties[index] for index, choice in enumerate(side_choices) if choice])
            if any(side_choices)
            else 0.0
        )
        internal_penalty = float(
            np.mean(
                [
                    MIXED_INTERNAL_PENALTY
                    for index, choice in enumerate(side_choices)
                    if choice and internal_side_flags[index]
                ]
            )
            if any(
                choice and internal_side_flags[index]
                for index, choice in enumerate(side_choices)
            )
            else 0.0
        )
        # La evidencia de la caja construida decide; las métricas por lado
        # sirven como auditoría y para penalizar movimientos o bordes internos.
        adjusted_score = float(candidate_score["score"] - movement_penalty - internal_penalty)
        row = {
            "side_choices": list(side_choices),
            "selected_sides": ["green_ransac" if choice else "blue_yolo" for choice in side_choices],
            "raw_score": float(candidate_score["score"]),
            "adjusted_score": adjusted_score,
            "geometry_score": float(candidate_score["geometry_score"]),
            "selected_side_scores": selected_side_scores.tolist(),
            "movement_penalty": movement_penalty,
            "internal_penalty": internal_penalty,
            "corners_crop_xy": candidate_box.round(3).tolist(),
        }
        candidate_rows.append(row)
        if best_candidate is None or adjusted_score > best_candidate["row"]["adjusted_score"]:
            best_candidate = {"box": candidate_box, "row": row}

    if best_candidate is None:
        best_candidate = {
            "box": order_points(initial_box),
            "row": {
                "side_choices": [0, 0, 0, 0],
                "selected_sides": ["blue_yolo"] * 4,
                "raw_score": float(blue["score"]),
                "adjusted_score": float(blue["score"]),
                "geometry_score": float(blue["geometry_score"]),
                "movement_penalty": 0.0,
                "internal_penalty": 0.0,
                "corners_crop_xy": order_points(initial_box).round(3).tolist(),
            },
        }

    blue_row = next(row for row in candidate_rows if row["side_choices"] == [0, 0, 0, 0])
    best_gain = float(best_candidate["row"]["adjusted_score"] - blue_row["adjusted_score"])
    best_choices = best_candidate["row"]["side_choices"]
    best_is_mixed = bool(any(best_choices) and best_choices != [1, 1, 1, 1])
    required_margin = MIXED_SCORE_MARGIN if best_is_mixed else CANDIDATE_SCORE_MARGIN
    use_best = bool(best_gain >= required_margin and best_choices != [0, 0, 0, 0])
    selected = best_candidate["box"] if use_best else order_points(initial_box)
    selected_choices = best_candidate["row"]["side_choices"] if use_best else [0, 0, 0, 0]
    selected_label = (
        "green_ransac"
        if selected_choices == [1, 1, 1, 1]
        else "mixed"
        if any(selected_choices)
        else "blue_yolo"
    )
    sides_not_worse = int(np.sum((green_side_scores - np.asarray(side_movement_penalties)) - blue_side_scores >= -0.08))
    sides_improved = int(np.sum((green_side_scores - np.asarray(side_movement_penalties)) - blue_side_scores >= 0.03))
    selection = {
        "route": "mixed_side_selection",
        "selected_candidate": selected_label,
        "selected_sides": best_candidate["row"]["selected_sides"] if use_best else ["blue_yolo"] * 4,
        "score_margin_required": required_margin,
        "full_candidate_score_margin": CANDIDATE_SCORE_MARGIN,
        "mixed_candidate_score_margin": MIXED_SCORE_MARGIN,
        "score_gain_best_minus_blue": best_gain,
        "sides_not_worse": sides_not_worse,
        "sides_improved": sides_improved,
        "blue_stable": blue_stable,
        "blue_external_guard": blue_external_guard,
        "external_guard_active": external_guard_active,
        "green_area_ratio_blue": green_area / blue_area,
        "green_sides_inside_blue": green_inside_flags,
        "green_internal_side_conflict": green_internal_side_conflict,
        "green_sides_allowed_by_external_guard": green_sides_allowed,
        "nested_green": nested_green,
        "internal_side_flags": internal_side_flags,
        "side_movement_penalties": side_movement_penalties,
        "candidate_count": len(candidate_rows),
        "top_candidates": sorted(candidate_rows, key=lambda row: row["adjusted_score"], reverse=True)[:5],
        "blue": blue,
        "green": green,
    }
    return selected, use_best and any(selected_choices), selection


def diagnostic_image(
    crop: np.ndarray,
    edges: np.ndarray,
    initial_box: np.ndarray,
    refined_box: np.ndarray,
    selection: dict | None = None,
    selected_box: np.ndarray | None = None,
) -> np.ndarray:
    overlay = crop.copy()
    cv2.polylines(overlay, [initial_box.astype(np.int32)], True, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(overlay, [refined_box.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    if selected_box is not None and selection is not None and selection.get("selected_candidate") == "mixed":
        cv2.polylines(overlay, [selected_box.astype(np.int32)], True, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(overlay, "azul: YOLO | verde: refinado", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    if selection is not None:
        selected_label = {
            "green_ransac": "verde RANSAC",
            "mixed": "mixto",
            "blue_yolo": "azul YOLO",
        }.get(selection["selected_candidate"], selection["selected_candidate"])
        if "blue" in selection and "green" in selection:
            decision_text = (
                f"elegido: {selected_label} | azul {selection['blue']['score']:.3f} "
                f"| verde {selection['green']['score']:.3f}"
            )
        else:
            decision_text = f"ruta legacy | elegido: {selected_label}"
        cv2.putText(
            overlay,
            decision_text,
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    target_height = 500
    panels = []
    for panel in (overlay, edge_bgr):
        width = round(panel.shape[1] * target_height / panel.shape[0])
        panels.append(cv2.resize(panel, (width, target_height), interpolation=cv2.INTER_AREA))
    return cv2.hconcat(panels)


def process_image(crop_path: Path, output_dir: Path, output_width: int) -> bool:
    metadata_path = crop_path.with_name(crop_path.name.replace("_yolo_crop.png", "_yolo_crop.json"))
    if not metadata_path.is_file():
        print(f"{crop_path.name}: falta {metadata_path.name}")
        return False
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None:
        print(f"No se pudo leer: {crop_path.name}")
        return False

    try:
        initial_box, metadata = load_seed(metadata_path, crop.shape[:2])
        refined_box, edges, legacy_was_refined = refine_corners(crop, initial_box)
        if USE_CANDIDATE_SELECTION:
            selected_box, was_refined, selection = select_preferred_candidate(
                crop, edges, initial_box, refined_box, float(metadata["confidence"])
            )
        else:
            selected_box, was_refined = refined_box, legacy_was_refined
            selection = {
                "route": "legacy",
                "selected_candidate": "green_ransac" if legacy_was_refined else "blue_yolo",
            }
        rectified = warp_card(crop, selected_box, output_width=output_width)
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        print(f"{crop_path.name}: no se pudo rectificar ({error})")
        return False

    base_name = crop_path.stem.replace("_yolo_crop", "")
    output_card = output_dir / f"{base_name}_card.png"
    output_debug = output_dir / f"{base_name}_refine_debug.png"
    output_metadata = output_dir / f"{base_name}_corners.json"
    cv2.imwrite(str(output_card), rectified)
    cv2.imwrite(str(output_debug), diagnostic_image(crop, edges, initial_box, refined_box, selection, selected_box))

    offset_x, offset_y, _, _ = metadata["crop_bounds_xyxy"]
    output_metadata.write_text(
        json.dumps(
            {
                "source_image": metadata["source_image"],
                "yolo_confidence": metadata["confidence"],
                "refinement_applied": was_refined,
                "selection": selection,
                "corners_crop_xy": selected_box.round(3).tolist(),
                "corners_source_xy": (selected_box + np.array([offset_x, offset_y])).round(3).tolist(),
                "output_size_wh": [output_width, round(output_width / CARD_RATIO)],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if selection["route"] == "mixed_side_selection":
        status = {
            "green_ransac": "RANSAC seleccionado",
            "mixed": "mixto seleccionado",
            "blue_yolo": "YOLO seleccionado (RANSAC no preferido)",
        }[selection["selected_candidate"]]
    else:
        status = "RANSAC" if was_refined else "semilla YOLO (sin ajuste confiable)"
    print(f"{crop_path.name}: {status} -> {output_card.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Refina bordes YOLO-OBB y rectifica cartas con homografía.")
    parser.add_argument("--input", type=Path, default=Path("img_segm_yolo"))
    parser.add_argument("--output", type=Path, default=Path("img_refined"))
    parser.add_argument("--limit", type=int, default=0, help="Máximo de recortes; 0 procesa todos.")
    parser.add_argument("--output-width", type=int, default=1000)
    parser.add_argument(
        "--legacy-refinement",
        action="store_true",
        help="Conserva siempre el comportamiento anterior y usa directamente el resultado de RANSAC.",
    )
    args = parser.parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input}")
    if args.output_width < 100:
        raise ValueError("--output-width debe ser al menos 100 píxeles.")

    sources = sorted(args.input.glob("*_yolo_crop.png"))
    sources = sources[: args.limit] if args.limit else sources
    args.output.mkdir(parents=True, exist_ok=True)
    global USE_CANDIDATE_SELECTION
    if args.legacy_refinement:
        USE_CANDIDATE_SELECTION = False
    successes = sum(process_image(source, args.output, args.output_width) for source in sources)
    print(f"\nCompletado: {successes}/{len(sources)} cartas rectificadas en {args.output}")


if __name__ == "__main__":
    main()
