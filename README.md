# Bioinformatics project
Cross over bioimaging and genomics - Feature extraction!

## Quick start

### Installation

### Data preparation


## Split dataset into training, validation and test sets
```
python src/split_train_test.py --testsizepercent 0.2 --valsizepercent 0.2
```

## Feature extraction

### Feature extraction from gene expression data
```
python src/genes_expression_analysis.py --cfg src\config\genes\conf.ini 
```
### Feature extraction from internal layers of CNN classifier
```
python src/images_feature_extraction.py 
```

## Concatenation of image and gene features

### Concatenation of image and gene features with PCA on images features
```
python src/feature_concatenation.py --cfg ./config/integration/conf.ini --n_principal_components 250 --plot_explained_variance
```

### Concatenation of image and gene features with gene copy ratio
```
python src/feature_concatenation.py --cfg ./config/integration/conf.ini --gene_copy_ratio 10
```

## Classification with images only

## Classification with genes only
### SVC
```
python src/genes_classifier.py --cfg src\config\genes\conf.ini --classification_method svm
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
```
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method linearsvc
```
### SGDClassifier
```
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method sgdclassifier
```
### Multi-Layered Perceptron
```
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method nn
```
### Multi-Layered Perceptron with data transformed with PCA
```
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method pca_nn
```


##Classification with class balancing

### SMOTE
```
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method <...> --balancing smote
```
### Random Upsampling
```
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method <...> --balancing random_upsampling
```
### SMOTEENN
```
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method <...> --balancing smoteenn
```
### Class Weights
```
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method <...> --balancing weights
```


