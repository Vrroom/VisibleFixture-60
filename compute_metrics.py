import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from piq import ssim, psnr, LPIPS
from tqdm import tqdm
_lpips_model = LPIPS(reduction='none')
lpips = lambda x, y, *args, **kwargs : _lpips_model(x, y)

def load_image_as_tensor(path: str) -> torch.Tensor:
    """Load a PNG, return [1, 3, H, W] float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t

def get_metric_fns():
    return [ssim, psnr, lpips], ['ssim', 'psnr', 'lpips']

def resize_long_side(t: torch.Tensor, long_side: int, exact_shape=None) -> torch.Tensor:
    """Resize [1,C,H,W] tensor so the long side equals long_side px, preserving aspect ratio.
    If exact_shape=(H,W) is given, resize to that exact size instead (for matching a reference)."""
    if exact_shape is not None:
        if t.shape[-2:] == torch.Size(exact_shape):
            return t
        return torch.nn.functional.interpolate(
            t, size=exact_shape, mode="bilinear",
            align_corners=False, antialias=True,
        ).clamp(0, 1)
    _, _, H, W = t.shape
    if max(H, W) == long_side:
        return t
    if H >= W:
        new_h = long_side
        new_w = max(1, int(round(long_side * W / H)))
    else:
        new_w = long_side
        new_h = max(1, int(round(long_side * H / W)))
    return torch.nn.functional.interpolate(
        t, size=(new_h, new_w), mode="bilinear",
        align_corners=False, antialias=True,
    ).clamp(0, 1)

@torch.no_grad()
def compute_metrics_pair(
    pred: torch.Tensor,
    gt: torch.Tensor,
    metric_fns,
    metric_names,
    image_resolution=960,
) -> dict:
    """
    pred, gt: [1, 3, H, W] tensors in [0, 1]. May differ in size. both are
    resized to image_resolution before comparison.
    Returns {ssim: float, psnr: float, lpips: float}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = pred.detach().float().clamp(0, 1).to(device)
    gt = gt.detach().float().clamp(0, 1).to(device)

    # Assert aspect ratios are similar (within 5% tolerance)
    pred_ar = pred.shape[-1] / pred.shape[-2]
    gt_ar = gt.shape[-1] / gt.shape[-2]
    assert abs(pred_ar - gt_ar) / max(pred_ar, gt_ar) < 0.05, \
        f"Aspect ratio mismatch: pred {pred.shape[-2]}x{pred.shape[-1]} (ar={pred_ar:.3f}) vs gt {gt.shape[-2]}x{gt.shape[-1]} (ar={gt_ar:.3f})"

    # Resize so that the long side is image_resolution px; match pred to gt's shape
    gt = resize_long_side(gt, image_resolution)
    pred = resize_long_side(pred, image_resolution, exact_shape=gt.shape[-2:])

    assert pred.shape == gt.shape, f"Shape mismatch after resize: {pred.shape} vs {gt.shape}"

    # Compute metrics 
    results = {}
    for fn, name in zip(metric_fns, metric_names):
        val = fn(pred, gt, data_range=1.0, reduction='none')
        results[name] = float(val.mean().item())

    return results

def list_of_dicts_to_df(records):
    records = list(records)
    if not records:
        return pd.DataFrame()
    cols = list(records[0].keys())
    if any(list(r.keys()) != cols for r in records):
        print("All items must have the same keys in the same order. but they do not. exiting ...")
        exit()
    return pd.DataFrame.from_records(records, columns=cols)

def main(export_dir: str):
    data_csv = os.path.join(export_dir, "data.csv")
    df = pd.read_csv(data_csv)
    print(f"Loaded {len(df)} rows from {data_csv}")

    metric_fns, metric_names = get_metric_fns()
    methods = ["tokenlight", "scribblelight"]
    results = []

    for i, row in tqdm(df.iterrows()):
        gt_path = os.path.join(export_dir, row["output_png"])
        gt = load_image_as_tensor(gt_path)

        entry = dict(
            idx=row["idx"],
            transition=row["transition"],
        )

        for method in methods:
            col = f"{method}_png"
            pred_path = os.path.join(export_dir, row[col])
            if not os.path.exists(pred_path):
                print(f"  WARNING: missing {pred_path}, skipping {method} for row {i}. This should not happen exiting ...")
                exit()

            pred = load_image_as_tensor(pred_path)
            metrics = compute_metrics_pair(
                pred, gt, metric_fns, metric_names,
                image_resolution=960,
            )
            for mn in metric_names:
                entry[f"{method}_{mn}"] = metrics[mn]

        results.append(entry)

    out_df = list_of_dicts_to_df(results)
    out_path = os.path.join(export_dir, "metrics.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics to {out_path}")

    # Print summary 
    for method in methods:
        cols = [f"{method}_{mn}" for mn in metric_names]
        print(f"\n{method} (all):")
        for c in cols:
            print(f"  {c}: {out_df[c].mean():.4f}")

        df_t1 = out_df[out_df["transition"] == "ON"]
        if not df_t1.empty:
            print(f"{method} (transition=ON only):")
            for c in cols:
                print(f"  {c}: {df_t1[c].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    main(args.export_dir)
