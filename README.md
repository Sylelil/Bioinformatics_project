# Bioinformatics project
Cross over bioimaging and genomics - Feature extraction!

## Quick start

### Installation

### Data preparation
Folder structure:
```
Bioinformatics_project
    assets
        original_genes_files
            normal
            tumor
        original_images_files
            normal
            tumor
         genes_manifest
            normal.json
            tumor.json
         images_manifest
            normal.json
            tumor.json         
```

```
python src/prepare.py 
```

## Split dataset into training, validation and test sets
```
python src/split_train_test.py --testsizepercent 0.2 --valsizepercent 0.2
```

## Feature extraction

### Generate the fine-tuned model
```
python src/images/fine_tuning.py
```

### Feature extraction from gene expression data
```
python src/genes_expression_analysis.py --cfg src/config/genes/conf.ini 
```
### Feature extraction from internal layers of CNN classifier
```
python src/images_feature_extraction.py 
```

### Feature extraction from internal layers of fine-tuned CNN classifier
```
python src/images_feature_extraction.py --with_fine_tuning
```

## Concatenation of image and gene features 
Insert in src/config/integration/conf.ini:
- use_features_images_only = False <br>
- apply_pca_to_features_images = False

```
python src/feature_concatenation.py --cfg src/config/integration/conf.ini
```

## Concatenation of image and gene features (with PCA)
Insert in src/config/integration/conf.ini:
- use_features_images_only = False <br>
- apply_pca_to_features_images = True <br>
- num_principal_components = 200

```
python src/feature_concatenation.py --cfg src/config/integration/conf.ini
```

## Prepare data for classification with images only 
Insert in src/config/integration/conf.ini:
- use_features_images_only = True <br>
- apply_pca_to_features_images = False <br>

```
python src/feature_concatenation.py --cfg src/config/integration/conf.ini
```

## Prepare data for classification with images only (with PCA)
Insert in src/config/integration/conf.ini:
- use_features_images_only = True <br>
- apply_pca_to_features_images = True <br>
- num_principal_components = 200

```
python src/feature_concatenation.py --cfg src/config/integration/conf.ini
```

## Classification with genes only
### SVC
```
python src/genes_classifier.py --cfg src/config/genes/conf.ini --classification_method svm
```

### Perceptron
```
python src/genes_classifier.py --cfg src\config\genes\conf.ini --classification_method perceptron
```

### SGDClassifier
```
python src/genes_classifier.py --cfg src\config\genes\conf.ini --classification_method sgd_classifier
```

## Classification of concatenated features

### LinearSVC
Insert in src/config/integration/conf.ini:
- use_features_images_only = False <br>
- apply_pca_to_features_images = True
- num_principal_components = 200

```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --classification_method linearsvc
```
### SGDClassifier
Insert in src/config/integration/conf.ini:
- use_features_images_only = False <br>
- apply_pca_to_features_images = True
- num_principal_components = 200
```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --classification_method sgdclassifier
```
### Multi-Layered Perceptron
Insert in src/config/integration/conf.ini:
- use_features_images_only = False <br>
- apply_pca_to_features_images = False

```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --classification_method nn
```
### Multi-Layered Perceptron with data transformed with PCA
Insert in src/config/integration/conf.ini:
- use_features_images_only = False <br>
- apply_pca_to_features_images = True
- num_principal_components = 200
```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --classification_method pcann
```

## Classification with images only

### LinearSVC
Insert in src/config/integration/conf.ini:
- use_features_images_only = True <br>
- apply_pca_to_features_images = True
- num_principal_components = 200
```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --classification_method linearsvc
```
### SGDClassifier
Insert in src/config/integration/conf.ini:
- use_features_images_only = True <br>
- apply_pca_to_features_images = True
- num_principal_components = 200

```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --classification_method sgdclassifier
```
### Multi-Layered Perceptron
Insert in src/config/integration/conf.ini:
- use_features_images_only = True <br>
- apply_pca_to_features_images = False

```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --classification_method nn
```
### Multi-Layered Perceptron with data transformed with PCA
Insert in src/config/integration/conf.ini:
- use_features_images_only = True <br>
- apply_pca_to_features_images = True
- num_principal_components = 200
```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --classification_method pcann
```

## Plot final results
```
python src/integration_classifier.py --cfg src/config/integration/conf.ini --plot_final_results
```