# Pokémon TCG Illustration Segmentation

An end-to-end Python pipeline that processes photographs of Pokémon Trading Card Game cards, rectifies each card, classifies its artwork layout, and extracts windowed illustrations with transparency.

## Dataset attribution

The input photographs originate from the public Kaggle dataset **[Pokémon TCG
Real Card Images](https://www.kaggle.com/datasets/ellimaaac/pokmon-tcg-real-card-images/data?select=1k)**
by **ellimaaac**. Its full distribution is approximately 47 GB, while this
project only needs the `1k/1k` image collection. Downloading the original
dataset in Colab solely to use that subset was impractical and could trigger
download-rate limits.

To make the notebook reproducible and lightweight, the project uses the public
derived dataset **[Pokémon TCG Real Card Images — 1K Subset](https://www.kaggle.com/datasets/rubielvelasquez/pokemon-tcg-real-card-images-1k-subset)**.
It is a copy of the original `1k/1k` files, created only to provide the required
subset directly to the pipeline. No labels, annotations, or image
transformations were added. The Colab notebook downloads this derivative
dataset, filters the `_1.jpg` input images, and copies them into `img/` before
running the pipeline.

Please consult the original dataset page for its terms, license, and attribution
requirements before redistributing the images or derived datasets. The derived
subset preserves attribution to **ellimaaac** and does not replace the original
source.

Pokémon and Pokémon Trading Card Game are trademarks of their respective owners. This repository is an independent technical project and is not affiliated with or endorsed by The Pokémon Company, Nintendo, Game Freak, or Creatures Inc.

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

## Trained models

The repository ships the three final (`best.pt`) weights used by the pipeline.
Each was produced by fine-tuning a pretrained **nano** (`n`) YOLO26 weight,
rather than using the base weight directly. The nano variant keeps inference
time and memory use low, which matters when the pipeline is also run on CPU.
The `s`, `m`, `l`, and `x` variants have greater capacity, but should be
compared under an equivalent validation setup before replacing these weights.

| Stage | Included final weight | Task and base weight | Training | Training images | Validation metrics for the best weight |
| --- | --- | --- | --- | --- | --- |
| Card localization | `models/card_obb/roboflow_obb_20260825_134802_best.pt` | OBB; `yolo26n-obb.pt` | 150 epochs, `imgsz=1024`, batch 4 | 157 training images (50 validation images). | Precision (B): 0.99855; recall (B): 0.98000; mAP50 (B): 0.99254; mAP50-95 (B): 0.98340. |
| Artwork-layout classification | `models/artwork_classifier/tipo_ilustracion_v1_best.pt` | Classification; `yolo26n-cls.pt` | 60 epochs, `imgsz=224`, batch 16 | 175 training images (37 validation and 37 test images). | Top-1 accuracy: 1.00000; top-5 accuracy: 1.00000; validation loss: 0.02817. |
| Illustration extraction | `models/illustration_segmentation/ilustracion_ventana_seg_20260825_191947_best.pt` | Segmentation; `yolo26n-seg.pt` | 150 epochs, `imgsz=640`, batch 8 | 124 training images (34 validation and 17 test images). | Mask (M): precision 1.00000, recall 0.99789, mAP50 0.99500, mAP50-95 0.98522. Box (B): precision 1.00000, recall 0.99789, mAP50 0.99500, mAP50-95 0.97525. |

The preceding values are the metrics stored in each checkpoint and correspond
to its **validation** split.

Segmentation is used only after a card is classified as `arte_ventana`: a
pixel-level mask is needed to create a transparent RGBA PNG. A bounding-box
detector would only return a rectangle and would include non-illustration
regions. In contrast, the first stage uses OBB because its four corners retain
the orientation of a rotated or perspective-photographed card, enabling later
rectification.

Each stage has a distinct role in turning a casual card photograph into a
usable illustration asset:

1. The input photograph is normalized while preserving its original visual
   content.
2. YOLO-OBB finds the card even when it is rotated or photographed at an
   angle, then produces a padded card crop.
3. RANSAC refines the four card edges and a perspective transform rectifies
   the crop into a front-facing card image.
4. The artwork-layout classifier separates standard windowed cards from
   full-art cards. For windowed cards, YOLO segmentation locates the artwork
   region, crops it, and writes it as an RGBA PNG with the predicted mask as
   its transparency channel. Full-art cards are retained in their own output
   folder because the illustration occupies the complete card face.

## Example results

The following two examples show the intended transformation: an initial
photograph on the left and the illustration asset produced by the pipeline on
the right.

| Windowed-art card (`arte_ventana`) | Full-art card (`arte_completo`) |
| --- | --- |
| **Initial photograph**<br><img src="starting_image_window.jpg" alt="Initial photograph of a windowed-art Pokémon card" width="280"> | **Initial photograph**<br><img src="starting_image_full.jpg" alt="Initial photograph of a full-art Pokémon card" width="280"> |
| **Extracted illustration**<br><img src="output_ilustration.png" alt="Extracted illustration from the windowed-art card" width="280"> | **Full-art output**<br><img src="output_ilustration_full.png" alt="Rectified full-art card output" width="280"> |

For the windowed-art example, the goal is the isolated illustration rather
than the full card. The full-art example illustrates the alternate branch:
the full card face is retained because its artwork extends across the card.

## Geometric refinement and crop selection

The YOLO-OBB detector provides a strong initial estimate of the card, but its
box is not guaranteed to coincide with the physical outer card boundary. This
is especially difficult in photographs containing protective slabs, glossy
reflections, holographic surfaces, perspective distortion, or low-contrast
card edges. In those conditions, the detector may follow an inner frame, while
an unconstrained edge detector may follow a line from the illustration,
printed text, an ability separator, or a reflection instead of the card edge.

This ambiguity matters downstream. The rectified image is resized before
artwork-layout classification, so a crop that is slightly too tight, or that
uses different physical boundaries on different sides, can change the visual
layout seen by the classifier. For windowed-art cards, the same geometry is
also used by the segmentation model to extract the illustration.

The `refine_segmentation_yolo.py` routine addresses this problem in several
stages:

1. It starts from the four YOLO-OBB corners and builds an adaptive edge map
   around the detected card.
2. It fits each side independently with RANSAC inside a narrow band around the
   YOLO estimate. The local band limits the risk of selecting unrelated lines
   elsewhere in the photograph.
3. It validates the RANSAC quadrilateral using the expected card aspect ratio,
   convexity, area change, displacement, and image bounds. Invalid or unstable
   refinements fall back to the YOLO box.
4. It compares candidates instead of accepting the RANSAC result automatically:

   - **Candidates:** each of the four sides can come from either YOLO or RANSAC.
     The routine evaluates the resulting 16 side-wise combinations, including
     the all-YOLO baseline and the all-RANSAC quadrilateral.
   - **Side evidence:** each candidate side is sampled from 10% to 90% of its
     length. Around every sample, the routine searches for nearby edge pixels
     within an 8-pixel radius. It records edge coverage and rewards a side when
     the evidence is close to the proposed line, continuous, strong in gradient,
     and separated by luminance contrast.
   - **Candidate score:** the side evidence is combined as 50% line closeness,
     25% continuity, 15% gradient strength, and 10% luminance contrast. The
     raw candidate score is then 90% mean side score plus 10% geometric
     consistency; the geometry component is based mainly on the card aspect
     ratio, with smaller contributions from side symmetry and image bounds.
     The adjusted score subtracts a movement penalty from selected RANSAC sides
     and, when the external-boundary guard is active, an additional penalty for
     an internal side.
   - **Decision:** the best adjusted candidate must improve on the YOLO baseline
     by at least 0.06. If it does not, the original YOLO quadrilateral is kept.
     This margin makes the refinement conservative: a new boundary must provide
     materially better evidence to replace the detector result.

5. When a high-confidence YOLO detection contains strong evidence for an
   external boundary, an additional guard prevents a RANSAC side that lies
   inside the YOLO box from replacing it. This side-level guard is important:
   a single internal line can occur even when the other RANSAC sides correctly
   expand the crop.

For example, for `126184781306_1`, the YOLO baseline scored 0.7378 and the
best mixed candidate (RANSAC top, right, and left sides plus the YOLO bottom
side) scored 0.8667 before penalties and 0.8051 after penalties. Its adjusted
gain over YOLO was 0.0674, which exceeded the 0.06 margin, so the mixed
quadrilateral was accepted. With a more restrictive 0.10 margin, the same
candidate would have been rejected even though its evidence was better; that
was the mechanism behind the previous regression toward `arte_ventana`.

The debug output for this same card is shown below. In the left panel, the blue
quadrilateral is the original YOLO estimate and the green quadrilateral is the
RANSAC refinement; the right panel shows the edge map used to evaluate the
candidate boundaries. The overlay makes the problem visible: the card boundary
is close to several strong lines, so the final decision must be made side by
side instead of replacing the complete YOLO box blindly.

<img src="126184781306_1_pre_refine_debug.png" alt="Debug view of YOLO and RANSAC refinement for card 126184781306_1" width="760">

The selected quadrilateral is finally rectified with a perspective transform
to the standard card output size. This preserves the reliable YOLO sides when
they are better, recovers valid external RANSAC sides when they are better,
and allows a mixed solution when the photograph does not have one uniformly
reliable boundary source. The selection metadata and debug image are saved
with each refined card so that the decision can be audited.

### Refinement example

The example below illustrates the problem and the resulting solution. The
input photograph contains perspective distortion and strong horizontal
reflections. The debug image overlays the initial YOLO estimate in blue and the
RANSAC refinement in green; its right panel shows the edge evidence used by the
routine. The final image is the selected quadrilateral after perspective
rectification.

| 1. Input photograph | 2. Edge and candidate diagnostic | 3. Rectified output |
| --- | --- | --- |
| <img src="refine_example_start.jpg" alt="Original photograph of a Pokémon card before refinement" width="280"> | <img src="refine_example_debug.png" alt="YOLO and RANSAC crop candidates with the edge map" width="280"> | <img src="refine_example_final.png" alt="Perspective-rectified Pokémon card after refinement" width="280"> |

This refinement is deliberately kept separate from artwork-layout
classification: it only decides which card boundary should be rectified. The
classifier then receives the normalized card image, and the segmentation
branch is run only when the card is classified as `arte_ventana`.

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
