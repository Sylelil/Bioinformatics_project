# Bioinformatics project
Cross over bioimaging and genomics - Feature extraction!

## Quick start

### Installation

### Data preparation


## Split dataset into training, validation and test sets
```
python src/split_train_test.py --testsizepercent 20 --valsizepercent 20
```

## Feature extraction

### Feature extraction from gene expression data
```
python src/load_genomics.py
```
### Feature extraction from internal layers of CNN classifier
```
python src/tiles_to_numpy.py
```


## Concatenation of image and gene features
```
python src/feature_concatenation.py --cfg ./config/integration/conf.ini
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