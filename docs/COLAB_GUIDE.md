# Google Colab Guide (for complete beginners)

## 1. What Colab is

Colab is a free Jupyter notebook that runs in your browser on Google's computers. You get a free GPU, which makes deep learning training 10-30x faster than a normal laptop. You write code in **cells** and run them one at a time.

You use Colab **only for training**. Your webcam data collection, the demo, the backend, and the frontend all run on your own laptop.

---

## 2. First-time setup (10 minutes)

1. Go to **colab.research.google.com** and sign in with a Google account.
2. Click **New notebook**. Rename it at the top (for example `03_train_cnn.ipynb`).
3. Turn on the GPU: **Runtime → Change runtime type → T4 GPU → Save**.
4. Check that it works. Type this in the first cell and press **Shift+Enter**:
   ```python
   import tensorflow as tf
   print(tf.__version__, tf.config.list_physical_devices("GPU"))
   ```
   You should see a GPU listed. If the list is empty, redo step 3.

### How cells work
- **Code cell**: Python. Run with Shift+Enter (runs and moves to the next cell).
- **Text cell**: notes (use `+ Text`).
- Lines starting with `!` run terminal commands (`!pip install ...`).
- Lines starting with `%` are notebook magics (`%cd folder`).
- Run cells **top to bottom**. If you get weird errors, use **Runtime → Restart session** and rerun from the top.

---

## 3. The 5 things that trip up beginners

1. **Sessions are temporary.** Everything in `/content` is deleted when the session ends (idle timeout or a few hours). Always save models and processed data to **Google Drive**.
2. **Drive is slow for many small files.** Copy your dataset from Drive to `/content` (local disk) before training, then save results back to Drive.
3. **Free GPU time is limited.** Do not leave sessions idle. Use `EarlyStopping`. Disconnect when done (**Runtime → Disconnect and delete runtime**).
4. **Install once per session.** Packages like MediaPipe must be reinstalled every new session (put the install in the first cell).
5. **After some `pip install`s Colab asks to restart.** Click **Restart session**, then run cells again *skipping the install cell*.

---

## 4. Project folders on Google Drive

Create this in your Drive (drive.google.com):

```
MyDrive/
└── sign-language/
    ├── kaggle/kaggle.json          # your Kaggle API key
    ├── processed/                  # cropped alphabet images (zipped) go here
    ├── custom/
    │   ├── alphabet.zip            # your own webcam alphabet crops
    │   └── phrases.zip             # your recorded (30,140) .npy files
    ├── models/                     # trained .keras files saved here
    └── reports/                    # plots
```

### Get your Kaggle API key
1. kaggle.com → profile picture → **Settings → API → Create New Token** (downloads `kaggle.json`).
2. Upload it to `MyDrive/sign-language/kaggle/`.

### Get your project code into Colab
Easiest: push your repo to GitHub, then in a notebook cell run
```python
!git clone https://github.com/<you>/sign-language-translator.git /content/repo
%cd /content/repo
```
(Private repo? Use a personal access token, or just zip `src/` and upload it to Drive, then `!unzip` it.)

Whenever you change `src/` in Cursor, `git push`, then in Colab `!git -C /content/repo pull`.

---

## 5. Starter cells for each notebook

### Cell 1 in EVERY notebook: setup

```python
# 1) Mount Google Drive (a popup asks permission)
from google.colab import drive
drive.mount('/content/drive')

# 2) Paths
DRIVE = "/content/drive/MyDrive/sign-language"

# 3) Code + packages
!git clone https://github.com/<you>/sign-language-translator.git /content/repo 2>/dev/null || git -C /content/repo pull
!pip install -q mediapipe==0.10.14 "numpy<2" scikit-learn seaborn tqdm
%cd /content/repo
import sys; sys.path.insert(0, "/content/repo")
```
If Colab says "restart session", restart, then run this cell again **from the `drive.mount` line, skipping the pip line**.

### Notebook 02: download and preprocess the alphabet dataset

```python
# Kaggle key
!mkdir -p ~/.kaggle
!cp {DRIVE}/kaggle/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and unzip to LOCAL disk (fast)
!pip install -q kaggle
!kaggle datasets download -d grassknoted/asl-alphabet -p /content/raw
!unzip -q /content/raw/asl-alphabet.zip -d /content/raw/asl
!ls /content/raw/asl
```
Find the folder that contains `A`, `B`, ..., `space`, `del` (usually `asl_alphabet_train/asl_alphabet_train`).

```python
import cv2, random, os
from pathlib import Path
from tqdm import tqdm
from src.preprocess import HandCropper
from src.config import ALPHABET_LABELS

SRC = Path("/content/raw/asl/asl_alphabet_train/asl_alphabet_train")  # adjust if needed
OUT = Path("/content/processed"); OUT.mkdir(exist_ok=True)
PER_CLASS = 1500
cropper = HandCropper(static_image_mode=True)

stats = {}
for label in ALPHABET_LABELS:
    files = sorted((SRC / label).glob("*.jpg"))
    random.seed(0); random.shuffle(files)
    (OUT / label).mkdir(exist_ok=True)
    ok = 0
    for f in tqdm(files[:PER_CLASS], desc=label):
        img = cv2.imread(str(f))
        crop, _ = cropper.crop(img)
        if crop is None: continue
        crop = cv2.resize(crop, (224, 224))
        cv2.imwrite(str(OUT / label / f"{ok}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        ok += 1
    stats[label] = ok
print(stats)

# Zip and save to Drive so you never redo this
!cd /content && zip -qr processed_alphabet.zip processed
!cp /content/processed_alphabet.zip {DRIVE}/processed/
```
This step is the slowest (roughly 20-40 minutes). Zip and save to Drive at the end so you only do it once.

### Notebooks 03/04: load processed data (every new session)

```python
!cp {DRIVE}/processed/processed_alphabet.zip /content/ && unzip -q /content/processed_alphabet.zip -d /content
!cp {DRIVE}/custom/alphabet.zip /content/ && unzip -q /content/alphabet.zip -d /content/custom_alphabet   # your own data
```
Then use `tf.keras.utils.image_dataset_from_directory` (batch 64, image_size 128 for CNN or 224 for MobileNet, `label_mode="int"`, fixed `class_names=ALPHABET_LABELS`) for train/val, combining Kaggle and your own crops.

Train, then **save straight to Drive**:
```python
from src.models import build_cnn
model = build_cnn(len(ALPHABET_LABELS), size=128)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

cbs = [
  tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
  tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
  tf.keras.callbacks.ModelCheckpoint(f"{DRIVE}/models/cnn.keras", save_best_only=True),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=40, callbacks=cbs)
```
`ModelCheckpoint` writes to Drive as it trains, so if the session dies you keep your best model.

MobileNet is the same but in two stages:
```python
from src.models import build_mobilenet, unfreeze_top
model = build_mobilenet(len(ALPHABET_LABELS), 224)
model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=8)

unfreeze_top(model, 30)
model.compile(tf.keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])  # recompile after changing trainable!
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=cbs)
```

Evaluate:
```python
import numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred = np.argmax(model.predict(test_ds), axis=1)
print(classification_report(y_true, y_pred, target_names=ALPHABET_LABELS))
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_true, y_pred), xticklabels=ALPHABET_LABELS, yticklabels=ALPHABET_LABELS, cmap="mako")
plt.savefig(f"{DRIVE}/reports/cm_cnn.png", dpi=150)
```

### Notebook 05: phrases (landmark sequences)

You record these on your **laptop** (needs a webcam) with `scripts/collect_phrases.py`. Zip `data/custom/phrases` and upload `phrases.zip` to Drive.

```python
!cp {DRIVE}/custom/phrases.zip /content/ && unzip -q /content/phrases.zip -d /content/phrases
import numpy as np, glob, os
from src.config import PHRASE_LABELS
X, y = [], []
for i, lab in enumerate(PHRASE_LABELS):
    for f in glob.glob(f"/content/phrases/**/{lab}/*.npy", recursive=True):
        X.append(np.load(f)); y.append(i)
X, y = np.array(X), np.array(y)
print(X.shape, np.bincount(y))    # X should be (N, 30, 140)
```
Split (by session if you can), train `build_sequence_model`, evaluate, save to `{DRIVE}/models/phrases_gru.keras`.

---

## 6. Bringing trained models back to your laptop

1. In Drive, open `sign-language/models/`, right-click each `.keras` file → **Download**.
2. Put them into `models/` in your repo. Also save `labels.json` there.
3. Use the same major version of TensorFlow/Keras locally as in Colab (`tf.__version__`). If loading fails with a version error, run `pip install tensorflow==<colab version>` locally.

---

## 7. Common problems and fixes

| Problem | Fix |
|---|---|
| `No GPU` in the device list | Runtime → Change runtime type → T4 GPU |
| `module 'mediapipe' has no attribute 'solutions'` | Install `mediapipe==0.10.14`, restart session |
| numpy / protobuf conflict after install | Restart session, do not reinstall, rerun the cells |
| Training suddenly stops | Session timed out. Reconnect, rerun setup, load your saved checkpoint from Drive |
| "Out of memory" | Lower batch size (64 → 32), or restart the session |
| Very slow data loading | You are reading from Drive. Copy to `/content` first |
| Drive mount popup did not appear | Click **Files** icon → **Mount Drive**, or rerun the cell |
| Kaggle `401 Unauthorized` | Regenerate `kaggle.json`, re-upload it |
| Accuracy 99% in Colab but bad on webcam | Domain gap. Add more of your own captured data, check that cropping is identical, test in different lighting |

---

## 8. Suggested schedule

| Day | Task |
|---|---|
| 1 | Cursor: M1-M2. Colab: setup practice, explore dataset |
| 2 | Cursor: M3, collect alphabet and phrase data. Colab: preprocess (notebook 02) |
| 3 | Colab: train CNN and MobileNet. Cursor: M4-M5 |
| 4 | Collect more phrase data, train GRU. Cursor: M6 desktop demo |
| 5 | Cursor/Antigravity: M7-M8 backend and frontend |
| 6 | Test, collect failing-sign data, retrain, polish UI |
| 7 | README, report, demo video |
