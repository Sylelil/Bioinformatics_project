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
```
python src/feature_concatenation.py --cfg ./config/integration/conf.ini
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
python src/integration_classifier.py --cfg ./config/integration/conf.ini --classification_method pcann
```