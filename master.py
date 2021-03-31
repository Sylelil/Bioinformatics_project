import os


def main():
    os.system('python src/genes/genes_select_svm_t_rfe.py')

    os.system('python src/images/fixed_feature_generator.py')


if __name__ == '__main__':
    main()
