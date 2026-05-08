# VisibleFixture-60

Evaluation data and metric computation code for the VisibleFixture-60 benchmark from
**TokenLight: Precise Lighting Control in Images using Attribute Tokens**.

## Data

Download the dataset from: `s3://suchatur-results/InteractiveRelighting/VisibleFixture60/`

```
VisibleFixture60/
  data.csv              # master index: maps each sample to its input/output/mask/prediction paths
  metrics.csv           # precomputed per-sample metrics
  compute_metrics.py    # recompute metrics yourself
  Input/                # input images (pairs of captures)
  Output/               # ground truth relit images
  Masks/                # light source masks
  TokenLight/           # our predictions
  ScribbleLight/        # ScribbleLight predictions
  Originals/            # original DNG files
  Exrs/                 # linear EXR files
```

Each row in `data.csv` describes one relighting pair. The `transition` column is either
`ON` (light turned on) or `OFF` (light turned off).

## Reproducing metrics

```
pip install torch numpy pandas piq pillow tqdm
python compute_metrics.py
```

This writes `metrics.csv` and prints per-method averages. Expected output:

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| ScribbleLight | 14.6356 | 0.5186 | 0.6114 |
| **TokenLight (ours)** | **20.0772** | **0.8476** | **0.2784** |

