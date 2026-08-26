"""Refina esquinas de recortes YOLO-OBB y rectifica las cartas por homografía."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from card_segmentation import CARD_RATIO, build_edge_map, order_points, refine_with_ransac, warp_card


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


def diagnostic_image(crop: np.ndarray, edges: np.ndarray, initial_box: np.ndarray, refined_box: np.ndarray) -> np.ndarray:
    overlay = crop.copy()
    cv2.polylines(overlay, [initial_box.astype(np.int32)], True, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.polylines(overlay, [refined_box.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay, "azul: YOLO | verde: refinado", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
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
        refined_box, edges, was_refined = refine_corners(crop, initial_box)
        rectified = warp_card(crop, refined_box, output_width=output_width)
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        print(f"{crop_path.name}: no se pudo rectificar ({error})")
        return False

    base_name = crop_path.stem.replace("_yolo_crop", "")
    output_card = output_dir / f"{base_name}_card.png"
    output_debug = output_dir / f"{base_name}_refine_debug.png"
    output_metadata = output_dir / f"{base_name}_corners.json"
    cv2.imwrite(str(output_card), rectified)
    cv2.imwrite(str(output_debug), diagnostic_image(crop, edges, initial_box, refined_box))

    offset_x, offset_y, _, _ = metadata["crop_bounds_xyxy"]
    output_metadata.write_text(
        json.dumps(
            {
                "source_image": metadata["source_image"],
                "yolo_confidence": metadata["confidence"],
                "refinement_applied": was_refined,
                "corners_crop_xy": refined_box.round(3).tolist(),
                "corners_source_xy": (refined_box + np.array([offset_x, offset_y])).round(3).tolist(),
                "output_size_wh": [output_width, round(output_width / CARD_RATIO)],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    status = "RANSAC" if was_refined else "semilla YOLO (sin ajuste confiable)"
    print(f"{crop_path.name}: {status} -> {output_card.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Refina bordes YOLO-OBB y rectifica cartas con homografía.")
    parser.add_argument("--input", type=Path, default=Path("img_segm_yolo"))
    parser.add_argument("--output", type=Path, default=Path("img_refined"))
    parser.add_argument("--limit", type=int, default=0, help="Máximo de recortes; 0 procesa todos.")
    parser.add_argument("--output-width", type=int, default=1000)
    args = parser.parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input}")
    if args.output_width < 100:
        raise ValueError("--output-width debe ser al menos 100 píxeles.")

    sources = sorted(args.input.glob("*_yolo_crop.png"))
    sources = sources[: args.limit] if args.limit else sources
    args.output.mkdir(parents=True, exist_ok=True)
    successes = sum(process_image(source, args.output, args.output_width) for source in sources)
    print(f"\nCompletado: {successes}/{len(sources)} cartas rectificadas en {args.output}")


if __name__ == "__main__":
    main()
