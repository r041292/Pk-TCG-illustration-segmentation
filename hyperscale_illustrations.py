"""Mejora ilustraciones de cartas y las hiperescalan de forma reproducible.

Las entradas predeterminadas son las salidas de la etapa 5.  Se conservan sus
rutas relativas en ``img_hyperscaled`` y, para las ilustraciones RGBA, el canal
alfa se escala por separado sin pasarlo por un modelo generativo.

Ejemplos:
    .\.venv\Scripts\python.exe hyperscale_illustrations.py --method opencv
    .\.venv\Scripts\python.exe hyperscale_illustrations.py --method realesrgan --device cpu
"""

from __future__ import annotations

import argparse
import csv
import sys
import types
from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_INPUT = Path("img_clasif")
DEFAULT_OUTPUT = Path("img_hyperscaled")
DEFAULT_MODEL = Path("models/RealESRGAN_x4plus.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=("opencv", "realesrgan"), default="opencv")
    parser.add_argument("--scale", type=int, default=4, help="Factor final; Real-ESRGAN admite 4.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tile", type=int, default=256, help="Tamaño de mosaico para Real-ESRGAN; 0 desactiva mosaicos.")
    parser.add_argument("--tile-pad", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda' o un dispositivo de PyTorch.")
    parser.add_argument("--limit", type=int, default=0, help="Máximo de imágenes; 0 procesa todas.")
    parser.add_argument("--overwrite", action="store_true", help="Permite reemplazar salidas existentes.")
    return parser.parse_args()


def input_files(input_dir: Path) -> list[Path]:
    """Devuelve solo los dos conjuntos de ilustraciones de la etapa 5."""
    complete = input_dir / "arte_completo"
    windows = input_dir / "arte_ventana" / "ilustraciones"
    files = [
        path
        for folder in (complete, windows)
        if folder.is_dir()
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(files)


def enhance_opencv(bgr: np.ndarray) -> np.ndarray:
    """Reduce ruido leve y recupera bordes sin correcciones globales agresivas."""
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 3, 3, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.0)
    return cv2.addWeighted(denoised, 1.22, blurred, -0.22, 0)


def resize_alpha(alpha: np.ndarray, scale: int) -> np.ndarray:
    """Conserva exactamente la cobertura binaria de la máscara segmentada."""
    return cv2.resize(alpha, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def load_realesrgan(model_path: Path, tile: int, tile_pad: int, device: str):
    """Carga Real-ESRGAN x4 y aplica compatibilidad para torchvision reciente."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No existe el peso de Real-ESRGAN: {model_path}. "
            "Descárguelo desde el release oficial y páselo con --model."
        )

    # BasicSR 1.4.2 aún importa este módulo eliminado por torchvision moderno.
    # El símbolo que necesita se mantiene en torchvision.transforms.functional.
    from torchvision.transforms.functional import rgb_to_grayscale

    compatibility_module = types.ModuleType("torchvision.transforms.functional_tensor")
    compatibility_module.rgb_to_grayscale = rgb_to_grayscale
    sys.modules.setdefault("torchvision.transforms.functional_tensor", compatibility_module)

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=False,  # CPU no admite correctamente la inferencia FP16.
        device=device,
    )


def process_image(image: np.ndarray, scale: int, method: str, upsampler) -> np.ndarray:
    bgr = image[:, :, :3]
    enhanced = enhance_opencv(bgr)
    if method == "opencv":
        result = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    else:
        result, _ = upsampler.enhance(enhanced, outscale=scale)
    if image.shape[2] == 4:
        alpha = resize_alpha(image[:, :, 3], scale)
        return np.dstack((result, alpha))
    return result


def main() -> None:
    args = parse_args()
    if args.scale < 1:
        raise ValueError("--scale debe ser al menos 1.")
    if args.method == "realesrgan" and args.scale != 4:
        raise ValueError("Real-ESRGAN_x4plus solo admite --scale 4.")
    if not args.input.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input}")

    sources = input_files(args.input)
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise FileNotFoundError("No encontré ilustraciones en arte_completo ni arte_ventana/ilustraciones.")

    upsampler = load_realesrgan(args.model, args.tile, args.tile_pad, args.device) if args.method == "realesrgan" else None
    rows: list[dict[str, object]] = []
    for index, source in enumerate(sources, start=1):
        relative = source.relative_to(args.input)
        destination = args.output / relative.with_suffix(".png")
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(sources)}] Omitido (ya existe): {relative}")
            continue
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"No se pudo leer una imagen RGB/RGBA válida: {source}")
        result = process_image(image, args.scale, args.method, upsampler)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), result, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"No se pudo escribir {destination}")
        rows.append(
            {
                "archivo_origen": str(relative),
                "archivo_salida": str(destination.relative_to(args.output)),
                "metodo": args.method,
                "escala": args.scale,
                "ancho_origen": image.shape[1],
                "alto_origen": image.shape[0],
                "canales": image.shape[2],
                "ancho_salida": result.shape[1],
                "alto_salida": result.shape[0],
            }
        )
        print(f"[{index}/{len(sources)}] {relative} -> {result.shape[1]}x{result.shape[0]}")

    if rows:
        report = args.output / f"resultado_hyperscaling_{args.method}.csv"
        with report.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCompletado: {len(rows)} imágenes. Reporte: {report}")


if __name__ == "__main__":
    main()
