"""Ejecuta de forma secuencial el procesamiento completo de cartas Pokémon.

Flujo: img -> img_pre -> img_segm_yolo -> img_refined -> img_clasif
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(label: str, script: str, arguments: list[str]) -> None:
    """Ejecuta una etapa y detiene el flujo si ésta falla."""
    command = [sys.executable, str(ROOT / script), *arguments]
    print(f"\n{'=' * 72}\nEtapa: {label}\nComando: {' '.join(command)}\n{'=' * 72}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Máximo de imágenes por etapa; 0 procesa todas.")
    parser.add_argument("--device", default="cpu", help="Dispositivo de YOLO, por ejemplo 'cpu' o '0'.")
    parser.add_argument("--obb-model", type=Path, help="Ruta al peso YOLO-OBB de detección de cartas.")
    parser.add_argument("--classifier-model", type=Path, help="Ruta al peso del clasificador de tipo de arte.")
    parser.add_argument("--segmentation-model", type=Path, help="Ruta al peso YOLO-seg de ilustraciones.")
    parser.add_argument(
        "--illumination-correction",
        action="store_true",
        help="Activa la corrección local de iluminación durante el preprocesamiento.",
    )
    parser.add_argument(
        "--white-balance",
        action="store_true",
        help="Activa el balance de blancos durante el preprocesamiento.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit no puede ser negativo.")

    preprocess_args = ["--limit", str(args.limit)]
    if args.illumination_correction:
        preprocess_args.append("--illumination-correction")
    if args.white_balance:
        preprocess_args.append("--white-balance")

    try:
        run_step("1/4 - Preprocesamiento", "pre-proc.py", preprocess_args)
        obb_args = ["--limit", str(args.limit), "--device", args.device]
        if args.obb_model:
            obb_args.extend(["--model", str(args.obb_model)])
        run_step("2/4 - Detección y recorte YOLO-OBB", "card_segmentation_yolo_obb.py", obb_args)
        run_step("3/4 - Refinamiento y rectificación", "refine_segmentation_yolo.py", ["--limit", str(args.limit)])
        classify_args = ["--limit", str(args.limit), "--device", args.device]
        if args.classifier_model:
            classify_args.extend(["--classifier-model", str(args.classifier_model)])
        if args.segmentation_model:
            classify_args.extend(["--segmentation-model", str(args.segmentation_model)])
        run_step("4/4 - Clasificación y extracción", "classify_and_extract_illustrations.py", classify_args)
    except subprocess.CalledProcessError as error:
        print(f"\nProceso detenido: la etapa falló con código {error.returncode}.", file=sys.stderr)
        raise SystemExit(error.returncode) from error

    print("\nPipeline completado. Revisa los resultados en img_clasif.")


if __name__ == "__main__":
    main()
