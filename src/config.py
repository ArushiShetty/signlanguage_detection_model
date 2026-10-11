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

SEQ_LEN = 30                  # 30 frames expected by trained Keras GRU layer
N_FEATURES = 140              # 63 left hand + 63 right hand + 14 pose

# Real-time smoothing tuned for comfortable 2-3s hold and deliberate pacing
ALPHA_WINDOW = 25              # window of recent frame votes
ALPHA_MIN_VOTES = 16           # hold same letter steadily for ~1.5 - 2.0s before committing
ALPHA_MIN_CONF = 0.55          # responsive confidence threshold
ALPHA_COOLDOWN_S = 2.2         # 2.2s calm buffer after committing before next letter starts!

PHRASE_STRIDE = 1              # evaluate on every frame
PHRASE_MIN_CONF = 0.50         # responsive phrase confidence
PHRASE_CONSECUTIVE = 4         # require 4 consistent frames to commit (~0.25s stable hold)
PHRASE_COOLDOWN_S = 2.5        # 2.5s calm buffer after committing a phrase!
