# Preprocessing
DESIRED_MAGNIFICATION = 10
TILE_SIZE = 224
SCALE_FACTOR = 32
OVERLAP = 0

USE_GPU = True

# Labels
NORMAL_LABEL = '0'
TUMOR_LABEL = '1'

BATCH_SIZE = 32

# Data augmentation
MULTIPLIER = 3

# Ottimizzazioni
NUM_SLIDES_PREFETCH = 3

# Modello
FINE_TUNED_MODEL_NAME = "FINE_TUNED_MODEL_NAME"
NUM_TRAINABLE_LAYERS = 34