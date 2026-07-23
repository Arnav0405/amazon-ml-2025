# Amazon ML Challenge 2025 — Product Price Prediction

Multimodal deep learning pipeline that predicts the **price of a product** from its
**image** plus a short **text catalog** (item name, description, bullet points, pack
size). Built for the **Amazon ML Challenge 2025** (hosted on Kaggle).

> **Best result:** the **CLIP + Self-Attention** model
> ([`src/dataset_model.ipynb`](src/dataset_model.ipynb)) scored **77% SMAPE Score** —
> the highest-scoring submission in this repo.

---

## The Problem

Given a large e-commerce catalog (~**20–25 GB**, images + text) — each product has:

- a **product image** (downloaded from a URL),
- a raw `catalog_content` text blob (item name, product description, several bullet
  points, and a `Value` + `Unit` pack size, e.g. `500 gram`),

…predict the product's **price**.

The competition is scored on **SMAPE Score** (based on Symmetric Mean Absolute Percentage
Error). Reported as a percentage, higher is better — **77%** here.

---

## Pipeline Overview

```
raw catalog (~20-25 GB)
      │
      ▼
1. building_img_dataset.ipynb   download images from URLs → rename to <sample_id>.jpg → dedupe → zip
      │
      ▼
2. dataset_cleaner.ipynb        parse catalog_content → Item Name / Description / Bullets / Value / Unit
      │                          strip emojis + weird chars, standardize units
      ▼
3. dataset_dsa.ipynb            EDA, log-transform price, 80/20 train/val split, batch into 3 parts
      │
      ├──────────────────────────────────────────────┐
      ▼                                                ▼
4a. dataset_model.ipynb  ⭐ WINNER              4b. diff_approach.ipynb  (alt)
    frozen CLIP + Self-Attention fusion             frozen CLIP + KNN(FAISS) features
    + MLP regressor  → 77%                           → XGBoost / LightGBM
      │                                                │
      ▼                                                ▼
5a. infer_test_.ipynb                          5b. infer_test.ipynb
    (attention model inference, 75k test)          (CLIP+KNN+LGBM inference, 75k test)
```

Supporting exploration: [`src/img_features.ipynb`](src/img_features.ipynb) extracts
1280-d MobileNetV2 image feature vectors (an earlier CNN-based direction).

---

## Data Preparation

### 1. Image download — `src/building_img_dataset.ipynb`
- Downloads every product image from its URL using
  [`src/utils.py`](src/utils.py) `download_images()` — a 100-worker
  `multiprocessing.Pool` over `urllib`.
- Renames files to `<sample_id>.jpg`, removes duplicate basenames and rows with no
  image, then zips each split.
- Runs for **train**, **val**, and **test** sets.
- [`src/check_allimgs.py`](src/check_allimgs.py) verifies every image opens (drops
  corrupt/truncated files, which the raw dump has plenty of).

### 2. Text cleaning — `dataset_cleaner.ipynb`
- The raw `catalog_content` is a newline blob. `parse_catalog_improved()` splits it into
  structured columns: **Item Name, Product Description, Value, Unit, Bullet Points**
  (missing fields → `NaN`).
- Strips emojis and non-ASCII punctuation via regex.
- **Standardizes units** with a lookup map (`fl oz`/`floz`/`fl. oz` → `fluid ounce`,
  `g`/`grams` → `gram`, etc.).
- Saves `cleaned_train.csv`.

### 3. EDA + splitting — `dataset_dsa.ipynb`
- **Log-transforms the target**: `log_price = log(price + 1)` — prices are heavily
  right-skewed, so the model regresses in log space and inverts with `expm1` at
  inference.
- 80/20 train/val split (`random_state=42`).
- Each split is **batched into 3 parts** (`part1/2/3`) — ~60k images ≈ 14 GiB, too large
  to hold at once, so training streams part-by-part with GPU memory freed between parts.

---

## ⭐ Winning Model — `src/dataset_model.ipynb`

**Frozen CLIP encoder → learnable multimodal self-attention fusion → MLP regressor.**

| Component | Detail |
|---|---|
| Backbone | `openai/clip-vit-base-patch32`, **frozen** (fp16), 512-d image + text embeds |
| Fusion | `MultimodalSelfAttention` — stacks `[image, text]` embeds, 8-head self-attention with residual + LayerNorm, so the model **learns per-sample how much to weight each modality** instead of naive concatenation |
| Head | 4-layer MLP `1024 → 512 → 256 → 128 → 1`, ReLU + dropout (0.3/0.2/0.1), final ReLU (price ≥ 0) |
| Target | `log_price`, loss = **MSE** (a `SMAPELoss` is also implemented in the notebook) |
| Optimizer | AdamW, weight decay 1e-2, grad-clip 1.0 |
| Schedule | `CosineAnnealingWarmRestarts` (T_0=5, T_mult=2), stepped every 500 batches |
| Params | **153M total / 1.74M trainable (1.14%)** — only the fusion + head train |
| Curriculum | 3 parts, decaying LR: part1 `1.29e-4` → part2 `5e-5` → part3 `1.18e-5`, 3 epochs each, batch size 4 |

**Why it wins:** the self-attention fusion learns cross-modal interactions (e.g. lean on
the image when the text is sparse) that a fixed concatenation head can't. CLIP's frozen
joint embedding space gives strong image+text features for free; only the tiny fusion +
regressor need training.

### Validation loss (MSE on log-price)

| Part | Start | Best |
|------|-------|------|
| Part 1 (3 ep) | 0.7413 | **0.6076** |
| Part 2 (3 ep) | 0.5722 | **0.5632** |
| Part 3 (3 ep) | 0.5658 | **0.5633** |

Final checkpoint saved as `model_final_3.pth` (regressor state dict only — CLIP is
frozen). Inference in [`src/infer_test_.ipynb`](src/infer_test_.ipynb) runs the 75k test
set, inverts with `np.expm1`, mean predicted price ≈ $17.

---

## Alternative Approach — `src/diff_approach.ipynb` + `src/infer_test.ipynb`

A two-stage **CLIP-features → gradient-boosting** pipeline:

1. Extract frozen CLIP image + text embeds, then **augment with KNN meta-features** via a
   FAISS index (`KNNFeatureAugmenter`, k=10) — for each product, features derived from its
   10 nearest neighbours in embedding space.
2. Save the combined `[text_embeds | img_embeds | text_knn | img_knn]` features to HDF5.
3. Train a **gradient-boosted regressor** — both **XGBoost** and **LightGBM** variants,
   with incremental training across the 3 parts and heavy regularization.

Inference (`infer_test.ipynb`) loads the LightGBM booster and scores all 75k test rows.
This line scored below the attention model, so the CLIP + Self-Attention notebook is the
final submission.

Also explored: `src/img_features.ipynb` — MobileNetV2 (ImageNet, global-avg-pool) →
1280-d image vectors per product (early CNN direction, superseded by CLIP).

---

## Repository Layout

```
.
├── dataset_cleaner.ipynb          # parse + clean catalog text, standardize units
├── dataset_dsa.ipynb              # EDA, log-transform price, split + batch
├── requirements.txt
└── src/
    ├── building_img_dataset.ipynb # download images from URLs, rename, dedupe, zip
    ├── check_allimgs.py           # drop corrupt / missing images
    ├── utils.py                   # multiprocessing image downloader
    ├── img_features.ipynb         # MobileNetV2 image feature extraction (exploratory)
    ├── dataset_model.ipynb        # ⭐ CLIP + Self-Attention model — 77% (WINNER)
    ├── diff_approach.ipynb        # alt: CLIP + KNN features + XGBoost / LightGBM
    ├── infer_test_.ipynb          # inference for the attention model
    └── infer_test.ipynb           # inference for the CLIP+KNN+LGBM model
```

> **Note:** the dataset, downloaded images, `.csv` splits, and `.pth`/model files are
> git-ignored (see `.gitignore`) — they're too large (~20–25 GB) to version. Regenerate
> them by running the notebooks in pipeline order.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The model notebooks additionally need (not in `requirements.txt`, install as needed):

```bash
pip install torch transformers peft faiss-cpu h5py lightgbm xgboost tensorflow
```

A CUDA GPU is assumed (`device_map="cuda"`, fp16 CLIP). Trained on an RTX 4060 Laptop GPU.

## Run Order

1. `src/building_img_dataset.ipynb` — download images
2. `dataset_cleaner.ipynb` — clean text → `cleaned_train.csv`
3. `dataset_dsa.ipynb` — log-transform, split, batch into parts
4. `src/dataset_model.ipynb` — train the winning model
5. `src/infer_test_.ipynb` — predict on the test set
