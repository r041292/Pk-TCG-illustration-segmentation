# Pokémon TCG Illustration Segmentation

An end-to-end Python pipeline that processes photographs of Pokémon Trading Card Game cards, rectifies each card, classifies its artwork layout, and extracts windowed illustrations with transparency.

## Pipeline

The complete workflow is executed by `run_pipeline.py`:

1. **Pre-processing** (`pre-proc.py`) applies EXIF orientation handling, conservative denoising, and optional white-balance or illumination correction.
2. **Card localization** (`card_segmentation_yolo_obb.py`) uses a YOLO oriented-bounding-box (YOLO-OBB) model to detect each card and produce a padded crop.
3. **Geometric refinement** (`refine_segmentation_yolo.py`) refines the detected edges with RANSAC and rectifies the card through a homography.
4. **Artwork classification and extraction** (`classify_and_extract_illustrations.py`) uses a YOLO classification model to distinguish `arte_completo` (full-art) from `arte_ventana` (windowed-art) cards. Windowed cards are then processed by a YOLO segmentation model; the predicted illustration mask is saved as an RGBA PNG.

The default flow is:

```text
img -> img_pre -> img_segm_yolo -> img_refined -> img_clasif
```

## Requirements

```bash
pip install -r requirements_pipeline.txt
```

`requirements.txt` additionally includes optional packages for the separate
illustration-upscaling utilities; it is not required for the main pipeline.

The final trained `.pt` weights used by the pipeline are included under `models/`:

- `models/card_obb/`: YOLO-OBB card detector.
- `models/artwork_classifier/`: artwork-layout classifier.
- `models/illustration_segmentation/`: YOLO segmentation model for windowed illustrations.

You can use these defaults directly, or override them with custom weights:

```bash
python run_pipeline.py \
  --obb-model /path/to/card_obb_model.pt \
  --classifier-model /path/to/artwork_classifier.pt \
  --segmentation-model /path/to/illustration_segmentation_model.pt
```

Use `--device 0` to run YOLO inference on the first available GPU, or leave the default `cpu`.

## Google Colab

[`notebooks/colab_pipeline_pokemon.ipynb`](notebooks/colab_pipeline_pokemon.ipynb) downloads the dataset, clones this repository, and runs the pipeline with its versioned final weights. It retains an optional, configurable Google Drive path if you later want to override those weights.

## Dataset attribution

The input photographs used by this project come from the public Kaggle dataset **[Pokémon TCG Real Card Images](https://www.kaggle.com/datasets/ellimaaac/pokmon-tcg-real-card-images/data?select=1k)** by **ellimaaac**. This pipeline uses its `1k` folder. Please consult the dataset page for its terms, license, and any attribution requirements before redistributing the images or derived datasets.

Pokémon and Pokémon Trading Card Game are trademarks of their respective owners. This repository is an independent technical project and is not affiliated with or endorsed by The Pokémon Company, Nintendo, Game Freak, or Creatures Inc.
