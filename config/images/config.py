from pathlib import Path

BASE_DIR = Path('.')

# Paths
selected_coords_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'coords'
selected_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'tiles'
results = BASE_DIR / 'results' / 'images'
images = BASE_DIR / 'datasets' / 'images'
normal_masked_images_dir = BASE_DIR / 'results' / 'images' / 'masked_images' / 'img_normal'
tumor_masked_images_dir = BASE_DIR / 'results' / 'images' / 'masked_images' / 'img_tumor'
low_res_normal_images_dir = BASE_DIR / 'results' / 'images' / 'low_res_images' / 'img_normal'
low_res_tumor_images_dir = BASE_DIR / 'results' / 'images' / 'low_res_images' / 'img_tumor'
normal_rand_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'rand_normal'
tumor_rand_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'rand_tumor'
splits_dir = BASE_DIR / 'assets' / 'data_splits'
extracted_features_train_dir = BASE_DIR / 'results' / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'train'
extracted_features_test_dir = BASE_DIR / 'results' / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'test'

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
