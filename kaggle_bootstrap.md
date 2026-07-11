# Kaggle bootstrap

Notebook cells to run **in order** to bring this repo up in a Kaggle
notebook, using the private GitHub repo and the private Kaggle dataset
that already holds the five raw signature-dataset folders (zipped at
top level).

Data stays mounted read-only at `/kaggle/input/<dataset-name>/`; all
outputs (checkpoints, metrics) go to `/kaggle/working/`. Nothing under
`data/raw` or `src/sigver/data/preprocessing.py` is touched by any of
this — the preprocessing pipeline is frozen at tag `preproc-v2-freeze`.

## 1. Read GITHUB_TOKEN from Kaggle Secrets

Add a Kaggle Secret with label `GITHUB_TOKEN` (Notebook -> Add-ons ->
Secrets) containing a GitHub personal access token with `repo` scope
for the private repo. Never paste the actual token value into a cell —
only read it via `UserSecretsClient`:

```python
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
GITHUB_TOKEN = user_secrets.get_secret("GITHUB_TOKEN")  # placeholder — real value lives only in Kaggle Secrets
```

## 2. Clone the private repo

```python
!git clone https://{GITHUB_TOKEN}@github.com/shrey-shetty/robust-signature-verification.git /kaggle/working/repo
%cd /kaggle/working/repo
```

## 3. Install the package

```python
# NEVER reinstall torch on Kaggle -- the preinstalled build is matched to
# the notebook's CUDA driver/runtime, and reinstalling it can silently
# break GPU support or waste the session downloading a multi-GB wheel.
#
# pyproject.toml declares no [project.dependencies], so plain
# `pip install -e .` is safe by itself -- it installs only the local
# `sigver` package and pulls in nothing else.
!pip install -e .
#
# requirements.txt DOES pin torch/torchvision (for local/non-Kaggle use)
# -- do NOT `pip install -r requirements.txt` verbatim on Kaggle. Install
# the other listed packages individually instead, skipping torch/torchvision:
!pip install numpy pandas scikit-learn matplotlib seaborn opencv-python Pillow PyYAML tqdm
```

## 4. Wire the data

The private Kaggle dataset must contain, after Kaggle's automatic
zip extraction, these five folder names directly under
`/kaggle/input/<dataset-name>/` (these are exactly what
`src/sigver/data/catalog.py` / `src/sigver/data/datasets.py` expect —
do not rename):

```
CEDAR
BHSig260-Bengali
BHSig260-Hindi
SignatureGPDSSyntheticOffLine4000
  \_ firmasSINTESISmanuscritas
signature_verification
  \_ full_org
  \_ full_forg
```

**Windows-zip nesting caveat:** if a folder was zipped on Windows by
right-clicking the folder itself (rather than its contents), the zip
can contain one extra wrapping directory, e.g.
`CEDAR/CEDAR/1/...` instead of `CEDAR/1/...`. Check this after Kaggle
extracts the dataset:

```python
import os
print(os.listdir("/kaggle/input/<dataset-name>/CEDAR")[:5])
# if this lists a single subfolder named "CEDAR" instead of writer-id
# folders like "1", "2", ... you have the extra nesting level -- see fix below
```

Two ways to point the scripts at the data (pick one):

**Option A — symlink** `data/raw` to the Kaggle input mount so no CLI
flags are needed and defaults behave exactly like local runs:

```python
import os
os.makedirs("data", exist_ok=True)
# adjust the nested path below if the Windows-zip caveat above applies,
# e.g. "/kaggle/input/<dataset-name>" -> "/kaggle/input/<dataset-name>/<dataset-name>"
!ln -s /kaggle/input/<dataset-name> data/raw
```

**Option B — `--raw-root` flag**, no symlink needed:

```
--raw-root /kaggle/input/<dataset-name>
```

(again adjusting for the extra nesting level if present).

## 5. Smoke test

Confirms data loading, device selection, and checkpointing work
end-to-end before committing to a long run. 2 epochs only — metrics are
meaningless, this is a plumbing check (see `configs/exp01_smoke.yaml`,
marked NOT-for-results).

```python
!python scripts/train_baseline.py --dataset cedar --epochs 2 \
    --raw-root /kaggle/input/<dataset-name> \
    --out /kaggle/working/exp01_smoke \
    --device cuda
```

(Omit `--raw-root` if you used the Option A symlink instead.)

## 6. Long runs: checkpoint resume across the 12h session cap

Kaggle sessions are capped at 12 hours. `train_baseline.py` writes
`last_checkpoint.pt` (model, optimizer state, epoch counter, best val
EER so far, history, and torch/numpy/python-random RNG state) every
epoch, distinct from `best_model.pt`. If a session gets cut off,
start a new session and resume:

```python
!python scripts/train_baseline.py --dataset cedar --epochs 30 \
    --raw-root /kaggle/input/<dataset-name> \
    --out /kaggle/working/siamese_smallcnn_cedar \
    --device cuda \
    --resume /kaggle/working/siamese_smallcnn_cedar/last_checkpoint.pt
```

`--out` must point at the same directory the interrupted run used
(persist it via a Kaggle Dataset/Notebook output between sessions,
since `/kaggle/working/` does not survive across independent sessions
by itself unless saved as output). Epoch numbering, best-model
selection (`min val_eer_all`), and `history.json` stay consistent
across the interrupt/resume boundary.
