from pathlib import Path

BASE_DIR = Path('.')

# config folder
genes_config_dir = BASE_DIR / 'config' / 'genes'  # Directory di configurazione per i geni

# assets folder
images_dir = BASE_DIR / 'assets' / 'images_files'    # Path alle WSI
genes_dir = BASE_DIR / 'assets' / 'genes_files'      # Path ai dati di genomica
split_data_dir = BASE_DIR / 'assets' / 'split_data'  # dati splittati
filename_splits_dir = BASE_DIR / 'assets' / 'filename_splits'  # filename splittati

# results folder

# results/genes
welch_t_results_dir = BASE_DIR / 'results' / 'genes' / 'welch_t'
svm_t_rfe_results_dir = BASE_DIR / 'results' / 'genes' / 'svm_t_rfe'
welch_t_selected_features_train = BASE_DIR / 'results' / 'genes' / 'welch_t' / 'selected_features' / 'train'
welch_t_selected_features_test = BASE_DIR / 'results' / 'genes' / 'welch_t' / 'selected_features' / 'test'
welch_t_selected_features_val = BASE_DIR / 'results' / 'genes' / 'welch_t' / 'selected_features' / 'val'
svm_t_rfe_selected_features_train = BASE_DIR / 'results' / 'genes' / 'svm_t_rfe' / 'selected_features' / 'train'
svm_t_rfe_selected_features_test = BASE_DIR / 'results' / 'genes' / 'svm_t_rfe' / 'selected_features' / 'test'
svm_t_rfe_selected_features_val = BASE_DIR / 'results' / 'genes' / 'svm_t_rfe' / 'selected_features' / 'val'

# results/images
images_results = BASE_DIR / 'results' / 'images'
extracted_features_train = BASE_DIR / 'results' / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'train'
extracted_features_test = BASE_DIR / 'results' / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'test'
extracted_features_val = BASE_DIR / 'results' / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'val'
selected_coords_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'coords'
selected_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'tiles'
normal_masked_images_dir = BASE_DIR / 'results' / 'images' / 'masked_images' / 'img_normal'
tumor_masked_images_dir = BASE_DIR / 'results' / 'images' / 'masked_images' / 'img_tumor'
low_res_normal_images_dir = BASE_DIR / 'results' / 'images' / 'low_res_images' / 'img_normal'
low_res_tumor_images_dir = BASE_DIR / 'results' / 'images' / 'low_res_images' / 'img_tumor'

# results/concatenated_results
# concatenated_results_dir = Path('C:\\') / 'Users' / 'rosee' / 'Downloads' / 'assets' / 'concatenated_results'
concatenated_results_dir = BASE_DIR / 'results' / 'concatenated_results'

