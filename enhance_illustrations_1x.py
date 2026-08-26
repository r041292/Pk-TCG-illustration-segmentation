"""Compara restauración de ilustraciones sin cambiar sus dimensiones.

Rutas disponibles:
* ``swinir``: reducción de ruido RGB 1x con el modelo oficial SwinIR.
* ``realesrgan_downsample``: Real-ESRGAN x4 seguido de reducción Lanczos al
  tamaño original; es una comparación experimental, no una mejora garantizada.

Conserva el alfa de las ilustraciones RGBA y escribe las variantes en
``img_enhanced_1x/<metodo>/`` sin alterar ``img_clasif`` ni ``img_hyperscaled``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SWINIR_SOURCE = PROJECT_ROOT / "third_party" / "SwinIR"
DEFAULT_SWINIR_MODEL = PROJECT_ROOT / "models" / "swinir" / "005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth"
DEFAULT_REALESRGAN_MODEL = PROJECT_ROOT / "models" / "RealESRGAN_x4plus.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("img_clasif"))
    parser.add_argument("--output", type=Path, default=Path("img_enhanced_1x"))
    parser.add_argument("--method", choices=("swinir", "realesrgan_downsample"), default="swinir")
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda' u otro dispositivo admitido por PyTorch.")
    parser.add_argument("--limit", type=int, default=0, help="Máximo de imágenes; 0 procesa todas.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--swinir-source", type=Path, default=DEFAULT_SWINIR_SOURCE)
    parser.add_argument("--swinir-model", type=Path, default=DEFAULT_SWINIR_MODEL)
    parser.add_argument("--swinir-tile", type=int, default=256, help="Mosaico SwinIR, múltiplo de 8; 0 procesa la imagen completa.")
    parser.add_argument("--swinir-overlap", type=int, default=32)
    parser.add_argument("--realesrgan-model", type=Path, default=DEFAULT_REALESRGAN_MODEL)
    parser.add_argument("--realesrgan-tile", type=int, default=256)
    parser.add_argument("--realesrgan-tile-pad", type=int, default=16)
    return parser.parse_args()


def input_files(input_dir: Path) -> list[Path]:
    folders = (input_dir / "arte_completo", input_dir / "arte_ventana" / "ilustraciones")
    return sorted(
        path
        for folder in folders
        if folder.is_dir()
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def install_basicsr_compatibility() -> None:
    """Compatibilidad de BasicSR 1.4.2 con torchvision reciente."""
    from torchvision.transforms.functional import rgb_to_grayscale

    module = types.ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = rgb_to_grayscale
    sys.modules.setdefault("torchvision.transforms.functional_tensor", module)


def load_realesrgan(args: argparse.Namespace):
    if not args.realesrgan_model.is_file():
        raise FileNotFoundError(f"No existe el peso Real-ESRGAN: {args.realesrgan_model}")
    install_basicsr_compatibility()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    network = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=str(args.realesrgan_model),
        model=network,
        tile=args.realesrgan_tile,
        tile_pad=args.realesrgan_tile_pad,
        pre_pad=0,
        half=False,
        device=args.device,
    )


def load_swinir(args: argparse.Namespace) -> torch.nn.Module:
    if not args.swinir_source.is_dir():
        raise FileNotFoundError(f"No existe el código oficial de SwinIR: {args.swinir_source}")
    if not args.swinir_model.is_file():
        raise FileNotFoundError(f"No existe el peso SwinIR de denoise: {args.swinir_model}")
    source = str(args.swinir_source)
    if source not in sys.path:
        sys.path.insert(0, source)
    from models.network_swinir import SwinIR

    model = SwinIR(
        upscale=1,
        in_chans=3,
        img_size=128,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="",
        resi_connection="1conv",
    )
    checkpoint = torch.load(args.swinir_model, map_location=args.device, weights_only=False)
    weights = checkpoint.get("params", checkpoint)
    model.load_state_dict(weights, strict=True)
    return model.eval().to(args.device)


def pad_to_window(image: torch.Tensor, window_size: int = 8) -> tuple[torch.Tensor, int, int]:
    """Aplica el mismo acolchado espejo usado por el script oficial SwinIR."""
    _, _, height, width = image.shape
    height_pad = (window_size - height % window_size) % window_size
    width_pad = (window_size - width % window_size) % window_size
    if height_pad:
        image = torch.cat([image, torch.flip(image, [2])], dim=2)[:, :, : height + height_pad, :]
    if width_pad:
        image = torch.cat([image, torch.flip(image, [3])], dim=3)[:, :, :, : width + width_pad]
    return image, height, width


@torch.inference_mode()
def swinir_enhance(bgr: np.ndarray, model: torch.nn.Module, device: str, tile_size: int, overlap: int) -> np.ndarray:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    tensor, height, width = pad_to_window(tensor)
    if not tile_size:
        result = model(tensor)
    else:
        if tile_size % 8:
            raise ValueError("--swinir-tile debe ser múltiplo de 8.")
        tile = min(tile_size, tensor.shape[2], tensor.shape[3])
        if tile % 8:
            tile -= tile % 8
        stride = tile - overlap
        if stride <= 0:
            raise ValueError("--swinir-overlap debe ser menor que --swinir-tile.")
        h_indices = list(range(0, tensor.shape[2] - tile, stride)) + [tensor.shape[2] - tile]
        w_indices = list(range(0, tensor.shape[3] - tile, stride)) + [tensor.shape[3] - tile]
        accumulation = torch.zeros_like(tensor)
        weights = torch.zeros_like(tensor)
        for top in h_indices:
            for left in w_indices:
                patch = model(tensor[:, :, top : top + tile, left : left + tile])
                accumulation[:, :, top : top + tile, left : left + tile].add_(patch)
                weights[:, :, top : top + tile, left : left + tile].add_(1)
        result = accumulation.div_(weights)
    result = result[:, :, :height, :width].squeeze(0).float().cpu().clamp_(0, 1).numpy()
    rgb_result = (result.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR)


def process_image(image: np.ndarray, method: str, model, args: argparse.Namespace) -> np.ndarray:
    bgr = image[:, :, :3]
    if method == "swinir":
        result = swinir_enhance(bgr, model, args.device, args.swinir_tile, args.swinir_overlap)
    else:
        enlarged, _ = model.enhance(bgr, outscale=4)
        result = cv2.resize(enlarged, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    if image.shape[2] == 4:
        return np.dstack((result, image[:, :, 3]))
    return result


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input}")
    sources = input_files(args.input)
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise FileNotFoundError("No se encontraron entradas en arte_completo ni arte_ventana/ilustraciones.")
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"Método: {args.method}; dispositivo: {args.device}; imágenes: {len(sources)}; "
        f"salida: {args.output}",
        flush=True,
    )
    model = load_swinir(args) if args.method == "swinir" else load_realesrgan(args)
    print("Modelo cargado. Iniciando inferencia…", flush=True)
    method_output = args.output / args.method
    rows: list[dict[str, object]] = []
    for index, source in enumerate(sources, start=1):
        relative = source.relative_to(args.input)
        destination = (method_output / relative).with_suffix(".png")
        if destination.exists() and not args.overwrite:
            print(f"[{index}/{len(sources)}] Omitido (ya existe): {relative}")
            continue
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"No se pudo leer como RGB/RGBA: {source}")
        print(f"[{index}/{len(sources)}] Procesando: {relative}", flush=True)
        start = time.perf_counter()
        result = process_image(image, args.method, model, args)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), result, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"No se pudo escribir {destination}")
        rows.append(
            {
                "archivo_origen": str(relative),
                "archivo_salida": str(destination.relative_to(args.output)),
                "metodo": args.method,
                "ancho": image.shape[1],
                "alto": image.shape[0],
                "canales": image.shape[2],
            }
        )
        elapsed = time.perf_counter() - start
        print(f"[{index}/{len(sources)}] Terminado en {elapsed:.1f} s: {relative}", flush=True)
    if rows:
        report = args.output / f"resultado_mejora_1x_{args.method}.csv"
        with report.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCompletado: {len(rows)} imágenes. Reporte: {report}")


if __name__ == "__main__":
    main()
