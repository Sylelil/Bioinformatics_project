from pathlib import Path

# Paths
BASE_DIR = Path('.')
selected_coords_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'coords'
selected_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'tiles'


# Labels
NORMAL_LABEL = '0'
TUMOR_LABEL = '1'

# Data augmentation
MULTIPLIER = 3

# Ottimizzazioni
NUM_SLIDES_PREFETCH = 3

# Modello
FINE_TUNED_MODEL_NAME = "FINE_TUNED_MODEL_NAME"
NUM_TRAINABLE_LAYERS = 34
