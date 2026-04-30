# Action Recognition for Real-Time Safety Monitoring

This repository contains a Two-Stream CNN‑LSTM action recognition pipeline built to classify human activities in short video clips into five safety-relevant categories (normal activity, fight, fall, crowd anomaly, panic/running). The core implementation is provided as a Jupyter notebook: `arjun-action-recognition1 (3).ipynb`.

**Key Features**
- Spatial feature extraction using a pre-trained ResNet18 backbone.
- Motion channel (frame difference) to capture velocity information.
- LSTM for temporal reasoning across frame sequences.
- Dataset utilities to build a unified CSV from a Videos folder and class mapping.

**Action classes (labels)**
- 0 — Normal Activity (sitting, standing, walking)
- 1 — Violent Interaction (fighting, hitting, struggling)
- 2 — Fall Detection (slips, collapses)
- 3 — Crowd Anomaly (sneaking or unusual patterns)
- 4 — Panic/Emergency (running, rapid movement)

**Repository contents**
- `arjun-action-recognition1 (3).ipynb` — Main notebook with data prep, mapping, and training workflow.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Recommended ignores for GitHub uploads.
- `LICENSE` — MIT license (replace author/year as needed).
- `CONTRIBUTING.md` — Short contribution guidelines.

Getting started
---------------
1. Install dependencies (recommended in a virtualenv):

```bash
pip install -r requirements.txt
```

2. Prepare data:
- Unzip `Videos.zip` (if provided) into a local folder named `Videos/`.
- Expected structure:

```
Videos/
├── sit/
├── stand/
├── hit/
└── ... (one folder per action category)
```
Uniform Temporal Sampling
To optimize processing, the system does not analyze every frame. Instead, it extracts exactly 12 frames spaced evenly across the duration of a clip. This ensures the model sees the beginning, middle, and end of an action regardless of video length.  

The Motion Channel
A key feature of this architecture is the Motion Map. By calculating the absolute difference between consecutive frames, the model highlights "hotspots" of high movement.

Fight/Panic: Shows intense, localized hotspots.  

Normal Activity: The motion map remains "cool" or quiet.  

Model Configuration
Input Resolution: 112x112  

Sequence Length: 12 frames  

Batch Size: 8  

Learning Rate: 3e-4  

Backbone: ResNet18 (modified for 4-channel input)

3. Build CSV metadata (from the notebook):
- The notebook contains `build_dataset_csv(video_root, output_csv_path)` which scans `Videos/`, maps folders to labels, and writes a `unified_dataset.csv`.

4. Run notebook cells in order:
- Open `arjun-action-recognition1 (3).ipynb` and follow the "Setup" cells to detect `VIDEO_ROOT` and generate `train.csv`, `val.csv`, and `test.csv`.

Notes and tips
--------------
- Training on video data can be slow and memory intensive; prefer a GPU runtime (Colab, Kaggle with GPU, or local machine with CUDA).
- For large datasets, keep `Videos` out of the Git repository — use cloud storage or release dataset separately.

Contact
-------
If you need help, open an issue describing the problem or desired enhancement.

License
-------
This project is released under the MIT License — see `LICENSE`.

