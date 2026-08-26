"""Clasifica cartas refinadas y extrae ilustraciones de las cartas con ventana.

El proceso recibe las cartas ya rectificadas en ``img_refined``. Primero las
clasifica como ``arte_completo`` o ``arte_ventana`` y conserva una copia de la
carta en ``img_clasif/<tipo>``. Para las de ventana usa el modelo YOLO-seg y
guarda la ilustración como PNG RGBA en ``img_clasif/arte_ventana/ilustraciones``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASSIFIER_MODEL = Path("models/artwork_classifier/tipo_ilustracion_v1_best.pt")
SEGMENTATION_MODEL = Path(
    "models/illustration_segmentation/ilustracion_ventana_seg_20260825_191947_best.pt"
)
EXPECTED_CLASSES = {"arte_completo", "arte_ventana"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("img_refined"))
    parser.add_argument("--output", type=Path, default=Path("img_clasif"))
    parser.add_argument("--classifier-model", type=Path, default=CLASSIFIER_MODEL)
    parser.add_argument("--segmentation-model", type=Path, default=SEGMENTATION_MODEL)
    parser.add_argument("--classification-imgsz", type=int, default=224)
    parser.add_argument("--segmentation-imgsz", type=int, default=640)
    parser.add_argument("--classification-conf", type=float, default=0.0)
    parser.add_argument("--segmentation-conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="'cpu', '0' u otro dispositivo admitido por PyTorch.")
    parser.add_argument("--limit", type=int, default=0, help="Máximo de cartas; 0 procesa todas.")
    return parser.parse_args()


def refined_cards(input_dir: Path) -> list[Path]:
    """Devuelve solamente las cartas finales y excluye JSON/diagnósticos."""
    cards = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS and path.stem.endswith("_pre_card")
    ]
    return sorted(cards)


def names_as_dict(names: dict | list) -> dict[int, str]:
    return {int(index): str(name) for index, name in enumerate(names)} if isinstance(names, list) else {int(k): str(v) for k, v in names.items()}


def classify_card(model: YOLO, image: Path, imgsz: int, conf: float, device: str) -> tuple[str, float]:
    result = model.predict(str(image), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if result.probs is None:
        raise RuntimeError("El modelo clasificador no devolvió probabilidades.")
    class_id = int(result.probs.top1)
    class_name = names_as_dict(result.names).get(class_id)
    if class_name not in EXPECTED_CLASSES:
        raise ValueError(f"Clase inesperada {class_name!r}; se esperaban {sorted(EXPECTED_CLASSES)}.")
    return class_name, float(result.probs.top1conf)


def best_mask(model: YOLO, image: Path, imgsz: int, conf: float, device: str) -> tuple[np.ndarray, float] | None:
    """Obtiene la máscara con mayor confianza, redimensionada a la carta fuente."""
    source = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"No se pudo leer {image}")
    result = model.predict(
        str(image), imgsz=imgsz, conf=conf, device=device, retina_masks=True, verbose=False
    )[0]
    if result.masks is None or result.boxes is None or len(result.masks) == 0:
        return None
    confidences = result.boxes.conf.detach().cpu().numpy()
    index = int(np.argmax(confidences))
    mask = result.masks.data[index].detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    height, width = source.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask, float(confidences[index])


def write_illustration(source_path: Path, mask: np.ndarray, destination: Path) -> tuple[int, int, int, int]:
    """Recorta el rectángulo mínimo de la máscara y conserva alfa transparente."""
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer {source_path}")
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("La máscara predicha no contiene píxeles.")
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    crop = image[top:bottom, left:right]
    alpha = (mask[top:bottom, left:right] * 255).astype(np.uint8)
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), rgba):
        raise OSError(f"No se pudo escribir {destination}")
    return left, top, right, bottom


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input}")
    for model_path, label in ((args.classifier_model, "clasificador"), (args.segmentation_model, "segmentación")):
        if not model_path.is_file():
            raise FileNotFoundError(f"No existe el modelo de {label}: {model_path}")

    sources = refined_cards(args.input)
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        raise FileNotFoundError(f"No encontré archivos '*_pre_card' en {args.input}")

    complete_dir = args.output / "arte_completo"
    window_dir = args.output / "arte_ventana"
    illustrations_dir = window_dir / "ilustraciones"
    complete_dir.mkdir(parents=True, exist_ok=True)
    window_dir.mkdir(parents=True, exist_ok=True)

    classifier = YOLO(str(args.classifier_model))
    segmenter = YOLO(str(args.segmentation_model))
    rows: list[dict[str, object]] = []
    for number, source in enumerate(sources, start=1):
        art_type, class_confidence = classify_card(
            classifier, source, args.classification_imgsz, args.classification_conf, args.device
        )
        card_destination = (complete_dir if art_type == "arte_completo" else window_dir) / source.name
        shutil.copy2(source, card_destination)
        row: dict[str, object] = {
            "archivo_origen": source.name,
            "tipo_arte": art_type,
            "confianza_clasificacion": f"{class_confidence:.6f}",
            "archivo_carta": str(card_destination.relative_to(args.output)),
            "archivo_ilustracion": "",
            "confianza_segmentacion": "",
            "recorte_xyxy": "",
            "estado": "ok",
        }
        if art_type == "arte_ventana":
            detection = best_mask(segmenter, source, args.segmentation_imgsz, args.segmentation_conf, args.device)
            if detection is None:
                row["estado"] = "sin_mascara_ilustracion"
            else:
                mask, segmentation_confidence = detection
                illustration_destination = illustrations_dir / f"{source.stem}_ilustracion.png"
                bounds = write_illustration(source, mask, illustration_destination)
                row["archivo_ilustracion"] = str(illustration_destination.relative_to(args.output))
                row["confianza_segmentacion"] = f"{segmentation_confidence:.6f}"
                row["recorte_xyxy"] = ",".join(map(str, bounds))
        rows.append(row)
        print(f"[{number}/{len(sources)}] {source.name}: {art_type} ({class_confidence:.3f}) - {row['estado']}")

    report_path = args.output / "resultado_clasificacion.csv"
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    windows = sum(row["tipo_arte"] == "arte_ventana" for row in rows)
    extracted = sum(bool(row["archivo_ilustracion"]) for row in rows)
    print(f"\nCompletado: {len(rows)} cartas; {windows} con ventana; {extracted} ilustraciones en {illustrations_dir}")
    print(f"Reporte: {report_path}")


if __name__ == "__main__":
    main()
