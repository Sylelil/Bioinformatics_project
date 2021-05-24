from pathlib import Path

BASE_DIR = Path('')

# assets folder
images_json_dir = BASE_DIR / 'assets' / 'images_manifest'    # Path manifest e json WSI
genes_json_dir = BASE_DIR / 'assets' / 'genes_manifest'      # Path manifest e json  dati di genomica
images_dir = BASE_DIR / 'assets' / 'images_files'    # Path alle WSI
genes_dir = BASE_DIR / 'assets' / 'genes_files'      # Path ai dati di genomica
original_images_dir = BASE_DIR / 'assets' / 'original_images_files'    # Path alle WSI
original_genes_dir = BASE_DIR / 'assets' / 'original_genes_files'      # Path ai dati di genomica
split_data_dir = BASE_DIR / 'assets' / 'split_data'  # dati splittati
filename_splits_dir = BASE_DIR / 'assets' / 'filename_splits'  # filename splittati

# results folder

# results/feature_extraction/genes
svm_t_rfe_results_dir = BASE_DIR / 'results' / 'feature_extraction' / 'genes' / 'svm_t_rfe'
svm_t_rfe_selected_features_train = BASE_DIR / svm_t_rfe_results_dir / 'selected_features' / 'train'
svm_t_rfe_selected_features_test = BASE_DIR / svm_t_rfe_results_dir / 'selected_features' / 'test'
svm_t_rfe_selected_features_val = BASE_DIR / svm_t_rfe_results_dir / 'selected_features' / 'val'

# results/feature_extraction/images
images_results = BASE_DIR / 'results' / 'feature_extraction' / 'images'
extracted_features_train = BASE_DIR / images_results / 'fixed_feature_generator' / 'extracted_features' / 'train'
extracted_features_test = BASE_DIR / images_results / 'fixed_feature_generator' / 'extracted_features' / 'test'
extracted_features_val = BASE_DIR / images_results / 'fixed_feature_generator' / 'extracted_features' / 'val'
selected_coords_dir = BASE_DIR / images_results / 'selected_tiles' / 'coords'

# results/concatenated_results
concatenated_results_dir = BASE_DIR / 'results' / 'concatenated_results'

# results/integration_classification
integration_classification_results_dir = BASE_DIR / 'results' / 'integration_classification'

# results/genes_classification
genes_classification_results_dir = BASE_DIR / 'results' / 'genes_classification'

# results/images_classification
images_classification_results_dir = BASE_DIR / 'results' / 'images_classification'