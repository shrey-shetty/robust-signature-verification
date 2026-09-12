"""Measure relative per-step training cost of each backbone.

Synthetic data only -- this isolates GPU compute and deliberately excludes
dataloading, which is shared across all arms. Because dataloading is a fixed
cost that does NOT scale with backbone size, the compute-only ratios below
OVERESTIMATE the true epoch-time ratio. Treat the extrapolations as an upper
bound, not a point estimate.
"""
import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sigver.models.backbones import build_backbone
from sigver.models.siamese import SiameseNetwork
from sigver.losses import ContrastiveLoss

# Anchor: measured ResNet-18 epoch time from that run's history.json.
RESNET18_EPOCH_SECONDS = 1719.9

BATCH = 32
WARMUP = 5
STEPS = 30
NAMES = ["resnet18", "efficientnet_b0", "vit_b_16"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")
if device.type != "cuda":
    print("WARNING: not on GPU -- these timings will not extrapolate.")

results = {}
for name in NAMES:
    backbone = build_backbone(name, embedding_dim=128, pretrained=True,
                              in_channels=1, l2_normalize=False,
                              input_norm="none")
    model = SiameseNetwork(backbone=backbone).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters())

    a = torch.rand(BATCH, 1, 150, 220, device=device)
    b = torch.rand(BATCH, 1, 150, 220, device=device)
    y = torch.randint(0, 2, (BATCH,), device=device).float()

    model.train()
    for _ in range(WARMUP):
        optimizer.zero_grad()
        ea, eb = model(a, b)
        criterion(ea, eb, y).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(STEPS):
        optimizer.zero_grad()
        ea, eb = model(a, b)
        criterion(ea, eb, y).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / STEPS * 1000.0

    results[name] = ms
    print(f"[{name:>16}] params={n_params:>12,}  {ms:7.1f} ms/step")

    del model, backbone, optimizer, a, b, y
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\n--- extrapolated epoch time (UPPER BOUND) ---")
base = results["resnet18"]
for name, ms in results.items():
    est = RESNET18_EPOCH_SECONDS * ms / base
    print(f"{name:>16}: {est/60:6.1f} min/epoch   x6 epochs = {est*6/3600:5.2f} h")
