"""Refina esquinas de recortes YOLO-OBB y rectifica las cartas por homografía."""

from __future__ import annotations

import argparse
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
CANDIDATE_SEARCH_RADIUS = 8
CANDIDATE_SIDE_START = 0.10
CANDIDATE_SIDE_END = 0.90


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


def _score_candidate(crop: np.ndarray, edges: np.ndarray, box: np.ndarray) -> dict:
    """Calcula el puntaje auditable de un cuadrilátero candidato."""
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


def select_preferred_candidate(
    crop: np.ndarray, edges: np.ndarray, initial_box: np.ndarray, refined_box: np.ndarray
) -> tuple[np.ndarray, bool, dict]:
    """Elige verde solo cuando mejora al azul con margen y de forma consistente."""
    blue = _score_candidate(crop, edges, initial_box)
    green = _score_candidate(crop, edges, refined_box)
    blue_side_scores = np.asarray([side["score"] for side in blue["sides"]], dtype=float)
    green_side_scores = np.asarray([side["score"] for side in green["sides"]], dtype=float)

    # RANSAC puede encontrar una línea interna fuerte (texto, separadores o
    # ilustración). Esa línea no debe ganar solo por tener más gradiente: cada
    # lado verde paga una penalización cuando se aleja demasiado del lado azul.
    blue_ordered = order_points(initial_box)
    green_ordered = order_points(refined_box)
    side_movement_ratios: list[float] = []
    side_movement_penalties: list[float] = []
    for blue_start, blue_end, green_start, green_end in zip(
        blue_ordered,
        np.roll(blue_ordered, -1, axis=0),
        green_ordered,
        np.roll(green_ordered, -1, axis=0),
    ):
        side_length = max(float(np.linalg.norm(blue_end - blue_start)), 1.0)
        endpoint_movement = (
            float(np.linalg.norm(green_start - blue_start))
            + float(np.linalg.norm(green_end - blue_end))
        ) / 2.0
        movement_ratio = endpoint_movement / side_length
        side_movement_ratios.append(movement_ratio)
        side_movement_penalties.append(0.18 * min(movement_ratio / 0.04, 1.0))

    adjusted_green_side_scores = green_side_scores - np.asarray(side_movement_penalties)
    side_delta = adjusted_green_side_scores - blue_side_scores

    # Un único lado no puede decidir el resultado: exigimos que al menos tres
    # lados no empeoren claramente y que dos aporten una mejora apreciable.
    sides_not_worse = int(np.sum(side_delta >= -0.08))
    sides_improved = int(np.sum(side_delta >= 0.03))
    score_gain = float(green["score"] - blue["score"])
    green_preferred = bool(
        score_gain >= CANDIDATE_SCORE_MARGIN
        and sides_not_worse >= 3
        and sides_improved >= 2
    )
    selected = refined_box if green_preferred else initial_box
    selection = {
        "route": "candidate_selection",
        "selected_candidate": "green_ransac" if green_preferred else "blue_yolo",
        "score_margin_required": CANDIDATE_SCORE_MARGIN,
        "score_gain_green_minus_blue": score_gain,
        "sides_not_worse": sides_not_worse,
        "sides_improved": sides_improved,
        "side_movement_ratios": side_movement_ratios,
        "side_movement_penalties": side_movement_penalties,
        "adjusted_green_side_scores": adjusted_green_side_scores.tolist(),
        "adjusted_side_delta_green_minus_blue": side_delta.tolist(),
        "blue": blue,
        "green": green,
    }
    return selected, green_preferred, selection


def diagnostic_image(
    crop: np.ndarray,
    edges: np.ndarray,
    initial_box: np.ndarray,
    refined_box: np.ndarray,
    selection: dict | None = None,
) -> np.ndarray:
    overlay = crop.copy()
    cv2.polylines(overlay, [initial_box.astype(np.int32)], True, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(overlay, [refined_box.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, "azul: YOLO | verde: refinado", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    if selection is not None:
        selected_label = "verde RANSAC" if selection["selected_candidate"] == "green_ransac" else "azul YOLO"
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
            selected_box, was_refined, selection = select_preferred_candidate(crop, edges, initial_box, refined_box)
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
    cv2.imwrite(str(output_debug), diagnostic_image(crop, edges, initial_box, refined_box, selection))

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
    if selection["route"] == "candidate_selection":
        status = "RANSAC seleccionado" if was_refined else "YOLO seleccionado (RANSAC no preferido)"
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
