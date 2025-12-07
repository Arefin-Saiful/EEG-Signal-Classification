# EEG Digit Classification (MindBigData 2023 MNIST-8B)

End-to-end EEG classification pipeline using the Hugging Face dataset **`DavidVivancos/MindBigData2023_MNIST-8B`** with configurable preprocessing and deep-learning models (EEGNet + optional ViT variants). :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

## What this project does
- Loads MindBigData2023 MNIST-8B from Hugging Face Datasets. :contentReference[oaicite:2]{index=2}
- Selects a **fixed subset of EEG channels** (12 channels by default). :contentReference[oaicite:3]{index=3}
- Applies EEG preprocessing:
  - Bandpass filter (3–30 Hz), DC removal, baseline correction, per-trial/channel z-normalization. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}
- Performs label filtering + remapping, and preprocesses in chunks using `datasets.map(...)`. :contentReference[oaicite:7]{index=7}
- Trains a classifier with:
  - Mixup augmentation (default alpha=0.4), class-weighted cross entropy, LR scheduling, early stopping. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
- Evaluates with accuracy, balanced accuracy, classification report, and confusion matrices. :contentReference[oaicite:11]{index=11}

## Dataset
This project uses:
- **MindBigData2023_MNIST-8B** via Hugging Face Datasets:
  - `load_dataset('DavidVivancos/MindBigData2023_MNIST-8B')` :contentReference[oaicite:12]{index=12}

> Note: The raw dataset provides EEG channels as columns like `FPz_0 ... FPz_T`, etc., plus metadata fields (e.g., `label`). :contentReference[oaicite:13]{index=13}

## Default configuration (important)
Key defaults in the code:
- Sampling rate: `FS = 128`
- Bandpass: `3–30 Hz`
- Selected channels (12): `["FPz","F8","FFT8h","FT7","FC5","FC1","FTT9h","FTT8h","T7","P7","P8","T8"]` :contentReference[oaicite:14]{index=14}
- Training: batch size 32, epochs 50, lr 1e-4, early stopping patience 10 :contentReference[oaicite:15]{index=15}
- Mixup: `MIXUP_ALPHA = 0.4` :contentReference[oaicite:16]{index=16}

## Models
### EEGNet (default)
Includes an EEGNet implementation (compact CNN for EEG). :contentReference[oaicite:17]{index=17}

### ViT variants (optional)
There are ViT-based model definitions (custom config + pretrained adaptation) included in the notebook/code, but EEGNet is the selected model in the training setup shown. :contentReference[oaicite:18]{index=18}

## Installation
### 1) Create environment
Recommended: Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
