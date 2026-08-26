"""Localiza cartas con el modelo YOLO-OBB y genera recortes para refinamiento.

Esta etapa no corrige perspectiva: conserva un recorte rectangular con margen,
las cuatro esquinas OBB y un diagnóstico. `refine_segmentation_yolo.py` usa
esos resultados para ajustar los bordes físicos y aplicar la homografía.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = Path(
    "models/card_obb/"
    "roboflow_obb_20260825_134802_best.pt"
)


def order_points(points: np.ndarray) -> np.ndarray:
    """Ordena los puntos como superior-izq., superior-der., inf.-der., inf.-izq."""
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    difference = np.diff(points, axis=1).ravel()
    ordered[1] = points[np.argmin(difference)]
    ordered[3] = points[np.argmax(difference)]
    return ordered


def select_best_obb(result) -> tuple[np.ndarray, float] | None:
    """Devuelve el OBB de mayor confianza de la única clase entrenada."""
    if result.obb is None or len(result.obb) == 0:
        return None
    confidences = result.obb.conf.detach().cpu().numpy()
    index = int(np.argmax(confidences))
    points = result.obb.xyxyxyxy[index].detach().cpu().numpy()
    return order_points(points), float(confidences[index])


def crop_bounds(points: np.ndarray, image_shape: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    """Calcula un recorte con margen, limitado a la imagen de entrada."""
    height, width = image_shape[:2]
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    pad = max(x_max - x_min, y_max - y_min) * margin
    left = max(0, int(np.floor(x_min - pad)))
    top = max(0, int(np.floor(y_min - pad)))
    right = min(width, int(np.ceil(x_max + pad)))
    bottom = min(height, int(np.ceil(y_max + pad)))
    if right <= left or bottom <= top:
        raise ValueError("El recorte YOLO no tiene área válida.")
    return left, top, right, bottom


def diagnostic_image(image: np.ndarray, points: np.ndarray, bounds: tuple[int, int, int, int], confidence: float) -> np.ndarray:
    debug = image.copy()
    cv2.polylines(debug, [points.astype(np.int32)], True, (255, 0, 0), 3, cv2.LINE_AA)
    left, top, right, bottom = bounds
    cv2.rectangle(debug, (left, top), (right - 1, bottom - 1), (0, 255, 255), 2)
    cv2.putText(debug, f"YOLO OBB {confidence:.3f}", (left, max(25, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return debug


def process_image(
    source: Path, output_dir: Path, model: YOLO, confidence_threshold: float, imgsz: int, margin: float, device: str
) -> bool:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        print(f"No se pudo leer: {source.name}")
        return False

    result = model.predict(image, imgsz=imgsz, conf=confidence_threshold, device=device, verbose=False)[0]
    detection = select_best_obb(result)
    if detection is None:
        print(f"{source.name}: sin detección con confianza >= {confidence_threshold:.2f}")
        return False

    points, confidence = detection
    bounds = crop_bounds(points, image.shape[:2], margin)
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right]
    stem = source.stem
    crop_path = output_dir / f"{stem}_yolo_crop.png"
    metadata_path = output_dir / f"{stem}_yolo_crop.json"
    debug_path = output_dir / f"{stem}_yolo_debug.png"
    local_points = points - np.array([left, top], dtype=np.float32)

    cv2.imwrite(str(crop_path), crop)
    cv2.imwrite(str(debug_path), diagnostic_image(image, points, bounds, confidence))
    metadata_path.write_text(
        json.dumps(
            {
                "source_image": source.name,
                "crop_image": crop_path.name,
                "confidence": round(confidence, 6),
                "crop_bounds_xyxy": [left, top, right, bottom],
                "obb_points_source_xy": points.round(3).tolist(),
                "obb_points_crop_xy": local_points.round(3).tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{source.name}: detectada ({confidence:.3f}) -> {crop_path.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Recorta cartas con YOLO-OBB para refinamiento posterior.")
    parser.add_argument("--input", type=Path, default=Path("img_pre"))
    parser.add_argument("--output", type=Path, default=Path("img_segm_yolo"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Máximo de imágenes; 0 procesa todas.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confianza mínima de YOLO.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=0.08, help="Margen relativo alrededor del OBB.")
    parser.add_argument("--device", default="cpu", help="'cpu', '0' u otro dispositivo admitido por PyTorch.")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"No existe el modelo: {args.model}")
    if not 0 <= args.margin <= 0.5:
        raise ValueError("--margin debe estar entre 0 y 0.5.")

    sources = sorted(path for path in args.input.iterdir() if path.suffix.lower() in VALID_EXTENSIONS)
    sources = sources[: args.limit] if args.limit else sources
    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    successes = sum(
        process_image(source, args.output, model, args.conf, args.imgsz, args.margin, args.device) for source in sources
    )
    print(f"\nCompletado: {successes}/{len(sources)} detecciones en {args.output}")


if __name__ == "__main__":
    main()
