from pathlib import Path

BASE_DIR = Path('.')

# assets folder
images_dir = BASE_DIR / 'assets' / 'images_files'    # Path alle WSI
genes_dir = BASE_DIR / 'assets' / 'genes_files'      # Path ai dati di genomica
split_data_dir = BASE_DIR / 'assets' / 'split_data'  # dati splittati
filename_splits_dir = BASE_DIR / 'assets' / 'filename_splits'  # filename splittati

# results folder

# results/feature_extraction/genes
welch_t_results_dir = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'welch_t'
svm_t_rfe_results_dir = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'svm_t_rfe'
welch_t_selected_features_train = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'welch_t' / 'selected_features' / 'train'
welch_t_selected_features_test = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'welch_t' / 'selected_features' / 'test'
welch_t_selected_features_val = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'welch_t' / 'selected_features' / 'val'
svm_t_rfe_selected_features_train = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'svm_t_rfe' / 'selected_features' / 'train'
svm_t_rfe_selected_features_test = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'svm_t_rfe' / 'selected_features' / 'test'
svm_t_rfe_selected_features_val = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'svm_t_rfe' / 'selected_features' / 'val'

# results/feature_extraction/images
images_results = BASE_DIR / 'results' / 'feature_extraction' / 'images'
extracted_features_train = BASE_DIR / 'results' / 'feature_extraction' / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'train'
extracted_features_test = BASE_DIR / 'results' / 'feature_extraction' / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'test'
extracted_features_val = BASE_DIR / 'results' / 'feature_extraction' / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'val'
selected_coords_dir = BASE_DIR / 'results' / 'feature_extraction' / 'images' / 'selected_tiles' / 'coords'
selected_tiles_dir = BASE_DIR / 'results' / 'feature_extraction' / 'images' / 'selected_tiles' / 'tiles'


# results/concatenated_results
concatenated_results_dir = BASE_DIR / 'results' / 'concatenated_results'
concatenated_pca_dir = BASE_DIR / 'results' / 'concatenated_pca'

# results/integration_classification
integration_classification_results_dir = BASE_DIR / 'results' / 'integration_classification'

