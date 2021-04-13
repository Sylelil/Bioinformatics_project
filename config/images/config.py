from pathlib import Path

# Paths
BASE_DIR = Path('.')
selected_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'coords'


# Labels
NORMAL_LABEL = '0'
TUMOR_LABEL = '1'

# Data augmentation
MULTIPLIER = 5

# Ottimizzazioni
NUM_SLIDES_PREFETCH = 3
