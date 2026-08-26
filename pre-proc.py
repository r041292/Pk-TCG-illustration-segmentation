"""Preprocesamiento inicial de fotografías de cartas Pokémon.

Esta etapa mejora la legibilidad y reduce diferencias de iluminación sin
detectar, recortar ni corregir la perspectiva de la carta: esas operaciones
pertenecen a la siguiente fase del proyecto.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def gray_world_white_balance(image: Image.Image) -> Image.Image:
    """Compensa dominantes de color sin forzar los colores propios de la carta."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    channel_means = rgb.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    gains = np.clip(gray_mean / np.maximum(channel_means, 1.0), 0.94, 1.06)
    corrected = np.clip(rgb * gains, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected)


def correct_local_illumination(image: Image.Image) -> Image.Image:
    """Compensa sombras suaves sobre L sin alterar los canales de color.

    La iluminación se estima con un desenfoque amplio. La división normaliza
    gradientes lentos (como una sombra de estuche) y se mezcla con el original
    para conservar la apariencia natural de la carta.
    """
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    luminance = lab[:, :, 0]
    sigma = max(rgb.shape[:2]) * 0.08
    illumination = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(luminance, np.maximum(illumination, 1), scale=float(np.mean(illumination)))
    blended = cv2.addWeighted(luminance, 0.45, normalized, 0.55, 0)
    local_contrast = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(blended)
    lab[:, :, 0] = cv2.addWeighted(blended, 0.70, local_contrast, 0.30, 0)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def reduce_camera_noise(image: Image.Image) -> Image.Image:
    """Reduce sólo ruido cromático fino, preservando contornos y texto.

    Un filtro bilateral pequeño no mezcla los dos lados de un borde; es mucho
    menos agresivo que suavizar y posteriormente aplicar enfoque.
    """
    rgb = np.asarray(image.convert("RGB"))
    denoised = cv2.bilateralFilter(rgb, d=3, sigmaColor=12, sigmaSpace=12)
    return Image.fromarray(denoised)


def preprocess_image(
    image: Image.Image,
    apply_illumination_correction: bool = False,
    apply_white_balance: bool = False,
) -> Image.Image:
    """Prepara la foto sin alterar de forma perceptible sus colores o detalle.

    La detección de carta e ilustración se realiza ahora con modelos de ML, por
    lo que el preprocesamiento no debe forzar contraste, saturación ni bordes.
    El balance de blancos y la corrección local quedan disponibles únicamente
    como opciones para fotografías excepcionalmente problemáticas.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    if apply_white_balance:
        image = gray_world_white_balance(image)
    if apply_illumination_correction:
        image = correct_local_illumination(image)
    return reduce_camera_noise(image)


def process_directory(
    input_dir: Path,
    output_dir: Path,
    limit: int,
    apply_illumination_correction: bool,
    apply_white_balance: bool,
) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

    files = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS)
    selected_files = files[:limit] if limit > 0 else files
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for source in selected_files:
        with Image.open(source) as image:
            result = preprocess_image(image, apply_illumination_correction, apply_white_balance)
            target = output_dir / f"{source.stem}_pre.png"
            result.save(target, format="PNG", optimize=True)
            written.append(target)
            print(f"Procesada: {source.name} -> {target.name}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesa fotos de cartas Pokémon.")
    parser.add_argument("--input", type=Path, default=Path("img"), help="Carpeta de imágenes de entrada.")
    parser.add_argument("--output", type=Path, default=Path("img_pre"), help="Carpeta de salida.")
    parser.add_argument("--limit", type=int, default=10, help="Máximo de imágenes (0 para todas).")
    parser.add_argument(
        "--illumination-correction", action="store_true",
        help="Activa corrección local de sombras en LAB-L (variante experimental).",
    )
    parser.add_argument(
        "--white-balance", action="store_true",
        help="Activa balance de blancos global (desactivado para preservar color).",
    )
    args = parser.parse_args()

    written = process_directory(
        args.input, args.output, args.limit, args.illumination_correction, args.white_balance
    )
    print(f"\nCompletado: {len(written)} imagen(es) en {args.output}")


if __name__ == "__main__":
    main()
