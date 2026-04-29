# Action Recognition for Real-Time Safety Monitoring

This project builds a video action recognition system for short CCTV-style clips. The main workflow is in `action_project_clean_final.ipynb`.

The notebook trains a 5-class safety monitoring classifier:

| Label | Class |
| --- | --- |
| `0` | `normal_activity` |
| `1` | `fight` |
| `2` | `fall` |
| `3` | `crowd_anomaly` |
| `4` | `running_panic` |

Important note: the provided dataset may not contain videos for `crowd_anomaly`. If class `3` has no samples, the model cannot truly learn that class.

## Project Structure

```text
action_recognition_project/
├── action_project_clean_final.ipynb   # Main notebook workflow
├── Videos.zip                         # Original dataset archive
├── data/
│   ├── Videos/                        # Extracted video folders
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── unified_dataset.csv
│   ├── best_model.pth                 # Saved best checkpoint
│   └── submission.csv                 # Generated predictions
├── dataset.py                         # Optional script pipeline helper
├── extract_frames.py                  # Optional frame extraction helper
├── model.py                           # Optional script model
├── train.py                           # Optional script training entry
├── predict.py                         # Optional script prediction entry
└── quick_test.py                      # Optional quick test helper
```

The notebook is the recommended entry point. The `.py` scripts are useful helpers, but they may not exactly match the final notebook model if you have been iterating in the notebook.

## Model Overview

The notebook uses a CNN-based video classifier:

1. Each video is sampled into a fixed number of frames.
2. Each RGB frame is resized.
3. A motion channel is computed using frame differences.
4. RGB + motion are stacked into a 4-channel input.
5. A pretrained ResNet18 extracts per-frame visual features.
6. A temporal classifier aggregates frame-level features.
7. The final layer predicts one of the 5 safety classes.

To keep training fast, the ResNet18 backbone can be frozen and only the temporal/classifier layers trained.

## Requirements

Install dependencies:

```bash
pip install torch torchvision opencv-python scikit-learn pandas matplotlib seaborn tqdm
```

For GPU training, install a CUDA-compatible PyTorch build from the official PyTorch install selector:

```text
https://pytorch.org/get-started/locally/
```

## Dataset Setup

Place `Videos.zip` in the project root:

```text
action_recognition_project/Videos.zip
```

The notebook extracts it automatically into:

```text
data/Videos/
```

Expected extracted folders may look like:

```text
data/Videos/
├── fall/
├── grab/
├── gun/
├── hit/
├── kick/
├── lying_down/
├── run/
├── sit/
├── sneak/
├── stand/
├── struggle/
├── throw/
└── walk/
```

## Label Mapping

The quality of the label mapping is very important. A recommended mapping is:

```python
FOLDER_TO_CLASS = {
    "sit":        {"label": 0, "class_name": "normal_activity"},
    "stand":      {"label": 0, "class_name": "normal_activity"},
    "walk":       {"label": 0, "class_name": "normal_activity"},

    "hit":        {"label": 1, "class_name": "fight"},
    "kick":       {"label": 1, "class_name": "fight"},
    "struggle":   {"label": 1, "class_name": "fight"},
    "grab":       {"label": 1, "class_name": "fight"},
    "throw":      {"label": 1, "class_name": "fight"},
    "gun":        {"label": 1, "class_name": "fight"},

    "fall":       {"label": 2, "class_name": "fall"},
    "lying_down": {"label": 2, "class_name": "fall"},
    "lyingdown":  {"label": 2, "class_name": "fall"},

    "run":        {"label": 4, "class_name": "running_panic"},
    "sneak":      {"label": 4, "class_name": "running_panic"},
}
```

If your assignment defines these actions differently, update the mapping before generating `unified_dataset.csv`.

## Running the Notebook Locally

1. Open `action_project_clean_final.ipynb`.
2. Run the setup/extraction cells.
3. Build `unified_dataset.csv`.
4. Split into `train.csv`, `val.csv`, and `test.csv`.
5. Run the imports/config cell.
6. Run frame extraction, transforms, dataset, and dataloader cells.
7. Run the model cell.
8. Run the training cell.
9. Run test evaluation and visualizations.
10. Generate `submission.csv`.

Recommended fast training settings:

```python
CONFIG["num_frames"] = 8
CONFIG["image_size"] = 112
CONFIG["batch_size"] = 8
CONFIG["num_epochs"] = 5
CONFIG["learning_rate"] = 1e-5
FAST_TRAINING = True
```

On Windows notebooks, keep:

```python
NUM_WORKERS = 0
```

This avoids multiprocessing hangs.

## Training Notes

If training is unstable or you see `NaN/Inf` warnings:

- Disable mixed precision/AMP.
- Use full precision training.
- Lower the learning rate to `1e-5`.
- Reinitialize the model.
- Delete any old `best_model.pth` created during a bad run.

Recommended stable setup:

```python
CONFIG["learning_rate"] = 1e-5
CONFIG["num_epochs"] = 5
scaler = None
FAST_TRAINING = True
```

If accuracy is poor but you do not want slow training:

- Fix the label mapping first.
- Keep ResNet frozen.
- Do not rely on class `3` if there are no class `3` videos.
- Evaluate plain model accuracy before using TTA or calibration.

If you have more time or a stronger GPU:

```python
FAST_TRAINING = False
CONFIG["num_epochs"] = 8
```

This allows limited fine-tuning and may improve accuracy, but it is slower.

## Inference on Random Test Videos

After training, use the notebook inference cell to sample random videos from `test_df` and show:

- true label
- predicted label
- confidence
- class probabilities
- representative frames

For datasets with no `crowd_anomaly` videos, you can prevent invalid class-3 predictions during inference:

```python
def predict_without_empty_class(probs):
    probs = probs.copy()
    probs[3] = -1.0
    return int(np.argmax(probs))
```

## GradCAM

The notebook includes GradCAM visualization to inspect which spatial regions influenced a prediction.

Because the model includes an LSTM, GradCAM backward on CUDA may require temporarily putting the model in training mode. The corrected GradCAM cell handles this by:

- setting the model to train mode for backward
- keeping ResNet BatchNorm layers in eval mode
- restoring eval mode after visualization

## Running on Google Colab

### Option 1: Upload Files Manually

1. Open Google Colab.
2. Upload `action_project_clean_final.ipynb`.
3. Upload `Videos.zip`.
4. Set runtime to GPU:

```text
Runtime -> Change runtime type -> T4 GPU
```

5. Install dependencies:

```python
!pip install opencv-python scikit-learn pandas matplotlib seaborn tqdm
```

PyTorch is usually preinstalled in Colab. Check GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

6. If `Videos.zip` is uploaded to `/content`, set:

```python
PROJECT_ROOT = "/content"
DATA_ROOT = "/content/data"
ZIP_PATH = "/content/Videos.zip"
VIDEO_ROOT = "/content/data/Videos"
```

Then run the notebook cells.

### Option 2: Use Google Drive

Mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Put the project folder in Drive, for example:

```text
/content/drive/MyDrive/action_recognition_project/
```

Then set:

```python
PROJECT_ROOT = "/content/drive/MyDrive/action_recognition_project"
DATA_ROOT = f"{PROJECT_ROOT}/data"
ZIP_PATH = f"{PROJECT_ROOT}/Videos.zip"
VIDEO_ROOT = f"{DATA_ROOT}/Videos"
```

Drive is convenient for saving checkpoints, but reading many videos from Drive can be slower than local Colab storage. For faster runs, copy the zip to `/content` first:

```python
!cp "/content/drive/MyDrive/action_recognition_project/Videos.zip" "/content/Videos.zip"
```

Then use `/content` paths.

### Colab Performance Tips

Use these settings first:

```python
CONFIG["num_frames"] = 8
CONFIG["image_size"] = 112
CONFIG["batch_size"] = 8
CONFIG["num_epochs"] = 5
CONFIG["learning_rate"] = 1e-5
FAST_TRAINING = True
```

In Colab/Linux, you may try:

```python
NUM_WORKERS = 2
```

If the dataloader crashes or becomes unstable, set:

```python
NUM_WORKERS = 0
```

## Troubleshooting

### No videos found

Check:

```python
print(VIDEO_ROOT)
print(os.listdir(VIDEO_ROOT))
```

If the zip extracted into nested folders, adjust `VIDEO_ROOT` or move the inner `Videos` folder to `data/Videos`.

### Some videos are not mapped

Check folder names:

```python
import os
print(os.listdir(VIDEO_ROOT))
```

Common issue: `lying_down` vs `LyingDown`. Normalize folder names and include both variants in `FOLDER_TO_CLASS`.

### Validation loss is NaN

Use:

```python
CONFIG["learning_rate"] = 1e-5
scaler = None
```

Reinitialize the model and remove old checkpoints:

```python
if os.path.exists(BEST_MODEL_PATH):
    os.remove(BEST_MODEL_PATH)
model = TwoStageActionModel(num_classes=CONFIG["num_classes"]).to(CONFIG["device"])
```

### Training is too slow

Keep:

```python
FAST_TRAINING = True
CONFIG["num_epochs"] = 5
CONFIG["num_frames"] = 8
CONFIG["image_size"] = 112
```

Avoid full backbone fine-tuning unless you have enough time and GPU memory.

## Outputs

The notebook can produce:

- `data/unified_dataset.csv`
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `data/best_model.pth`
- `data/submission.csv`
- confusion matrices
- classification reports
- GradCAM visualizations
- random test-set inference demos

## Limitations

- The model cannot learn classes that have no training videos.
- CCTV action labels can be subjective; incorrect folder-to-class mapping strongly reduces accuracy.
- Fast frozen-backbone training is efficient but may not reach the best possible accuracy.
- TTA and calibration are optional and can sometimes reduce accuracy if the base model is weak.

