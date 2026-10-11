# Sign Language Translator: Master Spec (for Cursor)

> Put this file in `docs/PROJECT_SPEC.md` in your repo. Tell Cursor: "Read docs/PROJECT_SPEC.md fully. Implement Milestone N only. Do not touch other milestones."

---

## 1. Goal

A real-time web app. The user shows ASL signs to a webcam. The app detects them within a fraction of a second and shows the translation as text, builds sentences, and can speak them aloud.

It recognizes two kinds of signs:

| Track | What | Input to model | Models |
|---|---|---|---|
| **A. Alphabet** | A-Z, `space`, `del` (static handshapes) | Cropped hand image (single frame) | **Custom CNN** and **MobileNetV2 (ImageNet transfer learning)** |
| **B. Phrases** | Hello, Thank you, Sorry, I love you, Please, Yes, No, Help, Good, Bye (+ `idle`) | Sequence of 30 frames of MediaPipe landmarks (hands + upper body) | **GRU sequence model** |

Why two tracks: letters are still images, so image models fit. "Thank you" and "Sorry" are **motions** (hand moving from chin, fist circling on chest). One image cannot capture motion, so phrases need a sequence of frames. Landmarks (numbers) train fast, are robust to background and lighting, and need far less data than raw video.

Computer-vision layer: **MediaPipe** (Hands for alphabet crops, Holistic for phrase landmarks).

> Verify every sign against a reference before recording data (Handspeak, Signing Savvy, ASL University). Sign descriptions in this doc are from general knowledge; confirm the exact motion.

---

## 2. Tech stack

- Python 3.10 or 3.11 (MediaPipe and TensorFlow both work reliably here)
- TensorFlow / Keras 3 (`.keras` model format)
- MediaPipe **pinned to `0.10.14`** (keeps the classic `mp.solutions` API that this spec uses)
- OpenCV, NumPy, scikit-learn, matplotlib, seaborn
- Backend: FastAPI + WebSocket (uvicorn)
- Frontend: Vite + React + TypeScript + Tailwind CSS + Framer Motion + lucide-react
- Training: Google Colab (free T4 GPU). Data collection: your laptop (needs webcam).

`requirements.txt`
```
tensorflow>=2.16
mediapipe==0.10.14
opencv-python
numpy<2
scikit-learn
matplotlib
seaborn
pandas
pillow
tqdm
fastapi
uvicorn[standard]
websockets
```

---

## 3. Repository structure

```
sign-language-translator/
├── README.md
├── requirements.txt
├── .cursorrules
├── .gitignore                       # data/, models/*.keras (optional), node_modules, .venv
├── docs/
│   ├── PROJECT_SPEC.md              # this file
│   └── COLAB_GUIDE.md
├── notebooks/                       # run in Colab
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocess_alphabet.ipynb
│   ├── 03_train_cnn.ipynb
│   ├── 04_train_mobilenet.ipynb
│   ├── 05_train_phrases.ipynb
│   └── 06_compare_models.ipynb
├── data/                            # gitignored
│   ├── raw/asl_alphabet/            # Kaggle download
│   ├── processed/alphabet/          # cropped hand images + npz
│   ├── custom/alphabet/<LABEL>/     # your own webcam crops
│   └── custom/phrases/<label>/<n>.npy   # each file = (30, 140)
├── models/
│   ├── cnn.keras
│   ├── mobilenet.keras
│   ├── phrases_gru.keras
│   └── labels.json
├── src/
│   ├── config.py
│   ├── preprocess.py                # hand crop for images
│   ├── landmarks.py                 # landmark feature extraction for sequences
│   ├── models.py                    # build_cnn, build_mobilenet, build_sequence_model
│   ├── train_utils.py               # datasets, callbacks, plots
│   ├── predictor.py                 # loads models, smoothing, sentence logic
│   └── realtime_demo.py             # OpenCV desktop demo
├── scripts/
│   ├── collect_alphabet.py          # capture your own hand crops per letter
│   └── collect_phrases.py           # record landmark sequences per phrase
├── backend/
│   ├── main.py                      # FastAPI + WebSocket
│   └── schemas.py
├── frontend/
│   ├── package.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── lib/ws.ts                # WebSocket client
│       ├── lib/speech.ts            # text-to-speech
│       ├── hooks/useWebcam.ts
│       ├── hooks/useSignSocket.ts
│       ├── components/
│       │   ├── Onboarding.tsx
│       │   ├── WebcamStage.tsx      # video + canvas overlay
│       │   ├── DetectedCard.tsx     # big animated letter/phrase
│       │   ├── ConfidenceRing.tsx
│       │   ├── TranslationPanel.tsx # sentence builder + speak/clear/copy
│       │   ├── ModeSwitch.tsx       # Alphabet | Phrases
│       │   ├── ModelSwitch.tsx      # CNN | MobileNet
│       │   ├── StatsBar.tsx         # FPS, latency
│       │   └── ReferenceChart.tsx   # ASL learning sidebar + practice mode
│       └── styles/index.css
└── reports/                         # confusion matrices, curves, screenshots
```

---

## 4. `src/config.py` (implement exactly)

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

ALPHABET_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del"]   # 28 classes
PHRASE_LABELS = ["idle", "hello", "thank_you", "sorry", "i_love_you",
                 "please", "yes", "no", "help", "good", "bye"]           # 11 classes
PHRASE_DISPLAY = {
    "hello": "Hello", "thank_you": "Thank you", "sorry": "Sorry",
    "i_love_you": "I love you", "please": "Please", "yes": "Yes",
    "no": "No", "help": "Help", "good": "Good", "bye": "Bye",
}

CNN_IMG_SIZE = 128
MOBILENET_IMG_SIZE = 224
HAND_CROP_PADDING = 0.25      # expand hand bbox by 25%

SEQ_LEN = 30                  # frames per phrase sample (~1 second)
N_FEATURES = 140              # 63 left hand + 63 right hand + 14 pose

# Real-time smoothing
ALPHA_WINDOW = 10             # last N frame predictions
ALPHA_MIN_VOTES = 7           # same letter in >= 7 of 10
ALPHA_MIN_CONF = 0.85
ALPHA_COOLDOWN_S = 0.8        # after committing a letter, wait before same letter again

PHRASE_STRIDE = 5             # run the sequence model every 5 frames
PHRASE_MIN_CONF = 0.85
PHRASE_CONSECUTIVE = 2        # same phrase this many times in a row
PHRASE_COOLDOWN_S = 1.5
```

---

## 5. Preprocessing modules

### 5.1 `src/preprocess.py` (alphabet crops)

Single source of truth for cropping. **Used identically in training data prep, collection scripts, and live inference.** If cropping differs between training and inference, accuracy collapses.

```python
class HandCropper:
    def __init__(self, static_image_mode: bool = True, max_hands: int = 1): ...
    def crop(self, bgr_image) -> tuple[np.ndarray | None, dict | None]:
        """
        Returns (crop_rgb_uint8_square, info) or (None, None) if no hand found.
        Steps: BGR->RGB, MediaPipe Hands, landmark min/max -> bbox,
        pad by HAND_CROP_PADDING, make it SQUARE around the center, clamp to image,
        return the crop (NOT yet resized).
        info = {"bbox": (x,y,w,h), "landmarks": [(x_norm,y_norm), ...21]}
        """
def resize_for_model(crop_rgb, size: int) -> np.ndarray:
    """cv2.resize to (size,size), float32, values stay in 0..255. Models normalize internally."""
```

Rules:
- Use `static_image_mode=True` for dataset prep and `False` for video.
- Models take raw 0-255 input; normalization is a `Rescaling` layer inside the model. Never normalize twice.

### 5.2 `src/landmarks.py` (phrase features)

Uses `mp.solutions.holistic`. For each frame return a `(140,)` float32 vector:

- Left hand: 21 landmarks x (x,y,z) = 63. Subtract the wrist (landmark 0), divide by max 2D distance from the wrist. Zeros if hand missing.
- Right hand: same, 63.
- Pose: landmarks `[0, 11, 12, 13, 14, 15, 16]` (nose, shoulders, elbows, wrists) x (x,y) = 14. Subtract the shoulder midpoint, divide by shoulder width. Zeros if missing. This tells the model where the hands are relative to the chin/chest, which is essential for Thank you / Sorry / Hello.

```python
import numpy as np
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]

def _norm_hand(lms):
    if lms is None:
        return np.zeros(63, dtype=np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
    pts -= pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1)) + 1e-6
    return (pts / scale).flatten()

def _norm_pose(lms):
    if lms is None:
        return np.zeros(14, dtype=np.float32)
    pts = np.array([[p.x, p.y] for p in lms.landmark], dtype=np.float32)
    center = (pts[11] + pts[12]) / 2
    width = np.linalg.norm(pts[11] - pts[12]) + 1e-6
    return ((pts[POSE_IDX] - center) / width).flatten()

def extract_features(results) -> np.ndarray:
    return np.concatenate([
        _norm_hand(results.left_hand_landmarks),
        _norm_hand(results.right_hand_landmarks),
        _norm_pose(results.pose_landmarks),
    ]).astype(np.float32)   # (140,)
```

Also expose a `LandmarkExtractor` class wrapping `mp.solutions.holistic.Holistic(model_complexity=1)` with `process(bgr_frame) -> (features, raw_results)`.

**Mirroring convention:** never flip frames before processing. The frontend mirrors the display with CSS only. Collection, training, and inference all see the raw camera frame.

---

## 6. Models (`src/models.py`)

All models accept **raw pixel range 0-255** (images) and contain their own preprocessing and augmentation layers. Augmentation layers are automatically inactive at inference.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def augment_block():
    return tf.keras.Sequential([
        layers.RandomRotation(0.08),          # about +-15 degrees
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.25, value_range=(0, 255)),
        layers.RandomContrast(0.25),
        # NO horizontal flip. Flipping changes handedness and sign meaning.
    ], name="augment")

def build_cnn(num_classes, size=128):
    inp = layers.Input((size, size, 3))
    x = augment_block()(inp)
    x = layers.Rescaling(1 / 255.0)(x)
    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="asl_cnn")

def build_mobilenet(num_classes, size=224):
    inp = layers.Input((size, size, 3))
    x = augment_block()(inp)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)      # MobileNetV2 expects [-1, 1]
    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(size, size, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="asl_mobilenet")

def unfreeze_top(model, n_layers=30):
    """Stage 2 fine-tuning: unfreeze last n_layers of the backbone, keep BatchNorm frozen."""
    base = [l for l in model.layers if isinstance(l, tf.keras.Model)][0]
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False

def build_sequence_model(num_classes, seq_len=30, n_feat=140):
    inp = layers.Input((seq_len, n_feat))
    x = layers.GaussianNoise(0.01)(inp)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.GRU(64)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="phrase_gru")
```

### Training recipes

| Model | Steps |
|---|---|
| CNN | Adam 1e-3, categorical/sparse CE, batch 64, up to 40 epochs, `EarlyStopping(patience=6, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.5, patience=3)` |
| MobileNetV2 | Stage 1: head only, Adam 1e-3, 8 epochs. Stage 2: `unfreeze_top(model, 30)`, Adam 1e-5, up to 15 epochs with early stopping |
| GRU phrases | Adam 1e-3, batch 32, up to 100 epochs, early stopping patience 12. Sequence augmentation (see 8.3) |

Class imbalance: use `class_weight` if counts differ by more than 1.5x.

Save `models/labels.json`:
```json
{"alphabet": ["A", "B", "..."], "phrases": ["idle", "hello", "..."]}
```
The label order in this file **must** match the training label order.

---

## 7. Datasets (Track A)

1. **ASL Alphabet (Kaggle: `grassknoted/asl-alphabet`)**: ~87k images, 29 folders (A-Z, space, del, nothing). Use up to **1,500 images per class** to keep preprocessing fast. Skip `nothing` (no-hand is handled by "MediaPipe found no hand").
2. **Your own webcam captures** (`scripts/collect_alphabet.py`): 150-300 crops per letter, varied lighting, backgrounds, distances, and both slight hand rotations. **This is the most important accuracy lever.** The Kaggle set has almost uniform backgrounds and the model will drop sharply on a real webcam without your own data.
3. Optional extra: search "ASL alphabet" on Roboflow Universe and Kaggle for varied-background sets and mix them in.

Preprocessing pipeline (`notebooks/02_preprocess_alphabet.ipynb`):
1. Walk class folders, sample up to 1,500 images per class.
2. `HandCropper.crop()` each image. Skip failures, log the detection success rate per class (report it).
3. Save square crops resized to 224x224 as uint8 into `data/processed/alphabet/<LABEL>/`. (CNN dataset resizes down to 128 on load.)
4. Merge with `data/custom/alphabet/` and split **train/val/test = 70/15/15**. Keep your own webcam images **out of training in part**: hold back at least 20% of your own images for test. That test score is the honest real-world score.

Note: J and Z are motion letters. Static models will be weaker on them. Say so in the report.

---

## 8. Phrase data (Track B)

### 8.1 Phrases and signs (ASL, verify against a reference)

| Label | Sign (approximate) | Nature |
|---|---|---|
| `idle` | Hands down or resting, random non-sign movement | class for "nothing happening" |
| `hello` | Open flat hand near forehead moves outward, like a salute | motion |
| `thank_you` | Flat fingertips at chin, hand moves forward/down toward the person | motion |
| `sorry` | Fist on chest, circular motion | motion |
| `i_love_you` | Thumb, index finger, and pinky extended, held up | static handshape |
| `please` | Flat hand on chest, circular motion | motion |
| `yes` | Fist nodding up and down | motion |
| `no` | Index and middle finger tap against thumb | motion |
| `help` | Fist on flat palm, both lift upward | motion (two hands) |
| `good` | Flat hand from chin moves down into other palm | motion (two hands) |
| `bye` | Open hand waves | motion |

Add more later by adding a label to `PHRASE_LABELS` and recording data.

### 8.2 Collection script `scripts/collect_phrases.py`

- Opens webcam (OpenCV), shows a live preview with a hand/pose skeleton.
- CLI: `python scripts/collect_phrases.py --label thank_you --samples 80`
- Flow per sample: on-screen "Get ready" for 1.5 s, then record exactly `SEQ_LEN` (30) frames, extract features each frame via `LandmarkExtractor`, save `(30, 140)` array to `data/custom/phrases/<label>/<timestamp>.npy`, short pause, repeat. `q` to quit, `r` to discard the last sample.
- Record **80-120 samples per phrase**, and **200+ for `idle`** (include hands at rest, hands moving randomly, scratching face, adjusting hair, mixing in other non-target gestures).
- Vary distance from camera, lighting, speed, small angle changes, and clothing. If possible, have 2-3 friends record too. This is what makes the model generalize beyond you.
- Keep `sessions` info: put each recording session in a subfolder or filename prefix (`s01_...`) so validation can be split by session, not randomly.

### 8.3 Sequence augmentation (in `train_utils.py`)

- Random time-stretch (resample 30 frames to 0.8x-1.2x speed, then pad/crop to 30)
- Gaussian jitter on features (sigma 0.01)
- Small random scale (0.9-1.1) on hand features
- Random frame dropout (zero 1-3 random frames)
- Optional: mirrored copy (negate x, swap left/right hand slots) only if you want left-handed support

### 8.4 Training (`notebooks/05_train_phrases.ipynb`)

Upload the `data/custom/phrases` folder to Google Drive, load all `.npy`, stack to `X (N, 30, 140)`, `y (N,)`, split by session, train, confusion matrix, save `models/phrases_gru.keras`.

---

## 9. Inference logic (`src/predictor.py`)

```python
class SignPredictor:
    def __init__(self):
        # load models/labels.json and the 3 models once, warm up each with a dummy input
        # create HandCropper(static_image_mode=False) and LandmarkExtractor()
    def process_frame(self, bgr_frame, mode: str, model_name: str) -> dict:
        """mode in {'alphabet','phrase'}, model_name in {'cnn','mobilenet'} (alphabet only)."""
    def reset(self): ...   # clear buffers and sentence
```

### Alphabet mode
1. `HandCropper.crop(frame)`. If None: clear the vote window, return `hand_detected=False`.
2. Resize to model size, predict, get `(label, conf, top3)`.
3. Push into a `deque(maxlen=ALPHA_WINDOW)`.
4. **Commit** a letter only if it appears >= `ALPHA_MIN_VOTES` times in the window, its mean confidence >= `ALPHA_MIN_CONF`, and it is not on cooldown (`ALPHA_COOLDOWN_S`). Then clear the window.
5. `space` appends " ", `del` removes the last character.

### Phrase mode
1. `LandmarkExtractor.process(frame)` gives a `(140,)` vector. Append it to a `deque(maxlen=SEQ_LEN)`.
2. Once the buffer holds 30 frames and at least one hand is visible (or was in the last 10 frames), run the model every `PHRASE_STRIDE` frames.
3. Commit a phrase if the top label is not `idle`, conf >= `PHRASE_MIN_CONF`, it was the top label `PHRASE_CONSECUTIVE` times in a row, and the cooldown passed. Then clear the buffer (prevents double firing).
4. Committed phrase appends its display text plus a space to the sentence.

Return format (same for both modes):

```json
{
  "hand_detected": true,
  "hands": [[[0.51, 0.62], "... 21 points"]],
  "bbox": [x, y, w, h],
  "prediction": {"label": "A", "display": "A", "confidence": 0.94},
  "top3": [{"label": "A", "confidence": 0.94}, {"label": "E", "confidence": 0.04}],
  "committed": null,
  "sentence": "HELLO ",
  "latency_ms": 21.3
}
```
`committed` is `{"token": "A"}` or `{"token": "Thank you", "kind": "phrase"}` only on the frame where something is added. Landmarks are normalized 0-1 in raw (non-mirrored) coordinates. In phrase mode, also include pose points for the overlay (optional).

---

## 10. Backend (`backend/main.py`)

FastAPI, one WebSocket endpoint `/ws`, plus `GET /health` and `GET /labels`.

- On startup: create one `SignPredictor` (load models once).
- Client sends JSON: `{"type":"frame","image":"<base64 jpeg>","mode":"alphabet","model":"cnn"}`, or `{"type":"reset"}`.
- Server decodes the JPEG with OpenCV, calls `process_frame`, replies with the JSON from section 9.
- **Backpressure:** the client sends the next frame only after receiving the previous response. The server never queues frames.
- Each WebSocket connection gets its own predictor state (buffers, sentence). Models are shared, buffers are per connection.
- Run inference off the event loop (`asyncio.to_thread`) so the socket stays responsive.
- CORS enabled for the Vite dev origin (`http://localhost:5173`).
- Run: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

Frame size the client sends: **320x240 JPEG, quality 0.7**, about 12-20 FPS. Small frames make it fast, and MediaPipe does not need more.

---

## 11. Frontend spec

### 11.1 Look and feel

- **Dark glassmorphism.** Background `#0A0C12` with two soft blurred radial gradients (cyan and violet) drifting slowly behind the UI.
- Accent gradient: `#22D3EE` (cyan) to `#8B5CF6` (violet). Success `#34D399`, warning `#FBBF24`, error `#F87171`.
- Panels: `bg-white/5`, `backdrop-blur-xl`, `border border-white/10`, `rounded-3xl`, soft shadow.
- Fonts: **Space Grotesk** (headings, big letters) and **Inter** (body) via Google Fonts, with system fallbacks.
- Motion (Framer Motion): letter card pops with spring on change, panels fade/slide in on mount, buttons scale on tap. Respect `prefers-reduced-motion`.
- Light/dark toggle. Mobile-friendly: on narrow screens the layout stacks (webcam on top, translation below).

### 11.2 Layout (desktop)

```
┌─────────────────────────────────────────────────────────────┐
│ Logo  SignSpeak        [Alphabet | Phrases]  [CNN | MobileNet]  FPS · ms  ☀/🌙 │
├───────────────────────────────────┬─────────────────────────┤
│                                   │  Detected                │
│        WEBCAM STAGE               │  ┌───────────────┐       │
│   video + glowing hand skeleton   │  │      A        │  ring │
│   status pill (no hand / ready)   │  └───────────────┘       │
│                                   │  Translation             │
│                                   │  "HELLO I LOVE YOU"      │
│                                   │  [🔊 Speak][Copy][Clear] │
├───────────────────────────────────┴─────────────────────────┤
│  ASL reference chart (collapsible) · Practice mode           │
└─────────────────────────────────────────────────────────────┘
```

### 11.3 Components

- **Onboarding**: full-screen intro, short "how it works", a big "Enable camera" button, friendly error if permission denied (with how to fix).
- **WebcamStage**: `<video>` mirrored with `transform: scaleX(-1)`, `<canvas>` overlay mirrored the same way. Draw the 21-point skeleton with neon glow (`shadowBlur`), a rounded bbox around the hand with the current label chip above it. Status pill: "Show your hand", "Detecting...", "Reconnecting...".
- **DetectedCard**: huge current prediction (letter or phrase text). Animates on change. Dimmed when confidence is low.
- **ConfidenceRing**: circular progress showing confidence, and a second thin ring showing "hold steady" progress toward committing. Color moves red, amber, green.
- **TranslationPanel**: the sentence, with the latest token highlighted briefly. Buttons: Speak (Web Speech API `speechSynthesis`), Copy, Clear (sends `{"type":"reset"}`), Backspace. Optional auto-speak toggle when a phrase commits.
- **ModeSwitch / ModelSwitch**: segmented controls. Changing mode resets buffers.
- **StatsBar**: client FPS (responses per second), server latency (`latency_ms`), connection dot.
- **ReferenceChart**: A-Z grid of small cards (letter and a text hint or image you add under `public/asl/`) plus a phrase list. **Practice mode**: shows a target letter/phrase, a green check animation when detected correctly, and a streak counter.

### 11.4 `useSignSocket` hook

- Connect to `ws://localhost:8000/ws` (configurable via `VITE_WS_URL`).
- Capture loop: draw video to a hidden 320x240 canvas, `toDataURL("image/jpeg", 0.7)`, send, **wait for response**, then schedule next frame with `requestAnimationFrame`.
- Auto reconnect with backoff. Expose `{prediction, hands, committed, sentence, latency, fps, connected}`.

### 11.5 Quality bar

- No layout shift, no flicker of the prediction (the server already smooths).
- Works in Chrome and Edge. Camera at `getUserMedia({video:{width:640,height:480}})`.
- Keyboard accessible controls, ARIA labels, sufficient contrast.

---

## 12. Milestones (feed to Cursor one at a time)

Run the code after each milestone before moving on.

**M1: Scaffolding.** Create the folder tree, `requirements.txt`, `.gitignore`, `src/config.py`, empty modules with docstrings.
Prompt: *"Read docs/PROJECT_SPEC.md. Implement Milestone M1 only."*

**M2: Preprocessing and landmarks.** `src/preprocess.py`, `src/landmarks.py`, a tiny test script that runs on a sample image and on the webcam and shows the crop/skeleton.
Prompt: *"Implement M2 exactly as in sections 5.1 and 5.2. Add scripts/test_crop.py that shows the live webcam with the hand crop in a second window."*

**M3: Collection scripts.** `scripts/collect_alphabet.py` (press a letter key to save a 224x224 crop into `data/custom/alphabet/<LETTER>/`, hold to burst-capture, on-screen counts per letter) and `scripts/collect_phrases.py` (section 8.2).
Then **you** collect data on your laptop.

**M4: Models and train utils.** `src/models.py` (section 6), `src/train_utils.py` (dataset loading from directories, `tf.data` pipeline, callbacks, confusion-matrix plot, sequence augmentation).

**M5: Notebooks** (run in Colab, see COLAB_GUIDE.md). Generate notebooks 01-06 following sections 7, 8.4, 6. Each notebook must be runnable top to bottom and save outputs to Google Drive.

**M6: Predictor and desktop demo.** `src/predictor.py` (section 9) and `src/realtime_demo.py` (OpenCV window with overlay, sentence bar, keys: `1`=alphabet, `2`=phrases, `m`=switch model, `c`=clear, `q`=quit). This is your checkpoint that everything works before building the web app.

**M7: Backend.** `backend/main.py`, `backend/schemas.py` (section 10). Include a small `scripts/ws_test.py` client that sends a test image.

**M8: Frontend.** Scaffold with Vite React TS, Tailwind, Framer Motion. Implement sections 11.1-11.4. Then polish (11.5). *This milestone is a great fit for Antigravity: ask its agent to run `npm run dev`, open the app in the browser, allow the fake webcam, and verify the layout, mode switch, and reconnect behavior.*

**M9: Test and improve.** Use it yourself, note which letters/phrases fail, record more data for those, retrain, repeat. Write `reports/`.

**M10: README and demo.** Write README with screenshots, GIF, architecture diagram, results table, limitations, future work. Optionally deploy (backend on Render/Railway/Hugging Face Spaces, frontend on Vercel). Camera access requires HTTPS in production.

---

## 13. Evaluation and report

For each model report: accuracy, macro F1, per-class precision/recall, confusion matrix, parameter count, `.keras` file size, inference ms per frame on CPU, and **accuracy on your own webcam test set**.

Expected comparison story: the CNN is smaller and trains from scratch. MobileNetV2 is usually more accurate and generalizes better thanks to ImageNet features but is heavier. Landmark GRU is small and background-independent.

Limitations to state honestly: single-signer bias, lighting, similar letters (M/N/S/T/A/E), J and Z motion, small phrase vocabulary, not a full sign language translator (no grammar, facial expressions).

Future work: more phrases via WLASL, transformer on landmarks, on-device browser inference with TF.js/ONNX, other sign languages (ISL).

---

## 14. Definition of done

- [ ] Alphabet mode recognizes all letters on your own webcam with >= 90% on your held-out own-data test
- [ ] Phrase mode reliably recognizes Hello, Thank you, Sorry, I love you (>= 90% in live tests)
- [ ] Letters and phrases feed a sentence that can be spoken
- [ ] End-to-end latency under ~150 ms per frame on a normal laptop
- [ ] Clean UI, no crashes on camera permission denied or backend disconnect
- [ ] README, report, confusion matrices, demo GIF

---

## 15. `.cursorrules` (save as a file named `.cursorrules` in the repo root)

```
You are helping build "SignSpeak", a real-time ASL translator. The source of truth is docs/PROJECT_SPEC.md. Read it before every task.

Rules:
- Implement only the milestone I name. Do not refactor or create files outside its scope.
- Python 3.10/3.11, TensorFlow/Keras 3 (.keras format), MediaPipe pinned to 0.10.14 (classic mp.solutions API).
- All hand cropping goes through src/preprocess.py HandCropper. All landmark features go through src/landmarks.py. Never duplicate this logic.
- Models take raw 0-255 image input and normalize internally. Never normalize twice.
- Never flip frames horizontally for processing. Mirror only visually in the frontend with CSS.
- No horizontal-flip augmentation for images.
- Label order must come from models/labels.json and src/config.py.
- Frontend: Vite + React + TypeScript + Tailwind + Framer Motion. Function components and hooks. No class components. Dark glassmorphism style per spec section 11.
- Keep functions small, typed, and documented. Add try/except around camera, socket, and model loading with clear error messages.
- After writing code, tell me the exact command to run it and what I should see.
- If something in the spec is ambiguous, pick the simplest option and tell me what you chose.
```

---

## 16. Local setup commands

```bash
git clone <your-repo> && cd sign-language-translator
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# collect data
python scripts/collect_alphabet.py
python scripts/collect_phrases.py --label thank_you --samples 80

# desktop demo (after models are in models/)
python -m src.realtime_demo

# backend
uvicorn backend.main:app --port 8000 --reload

# frontend
cd frontend && npm install && npm run dev
```
