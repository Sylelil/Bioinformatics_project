import argparse
import os
import sys
from os import path
from pathlib import Path
from images import preprocessing, slide_info, utils
from images.features_extraction_methods.fine_tuning import fine_tuning
from images.features_extraction_methods.fixed_feature_generator import fixed_feature_generator
from common import split_data

USE_GPU = True
BASE_DIR = Path('..') / '..'

def main():
    if not USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Parse arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--method',
                        help='Feature extraction method',
                        choices=['fine_tuning', 'fixed_feature_generator'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    # Paths
    results = BASE_DIR / 'results' / 'images'
    normal_images = BASE_DIR / 'datasets' / 'images' / 'normal'
    tumor_images = BASE_DIR / 'datasets' / 'images' / 'tumor'

    selected_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'coords'
    normal_selected_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'normal_coords'
    tumor_selected_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'tumor_coords'

    normal_masked_images_dir = BASE_DIR / 'results' / 'images' / 'masked_images' / 'img_normal'
    tumor_masked_images_dir = BASE_DIR / 'results' / 'images' / 'masked_images' / 'img_tumor'

    low_res_normal_images_dir = BASE_DIR / 'results' / 'images' / 'low_res_images' / 'img_normal'
    low_res_tumor_images_dir = BASE_DIR / 'results' / 'images' / 'low_res_images' / 'img_tumor'

    normal_rand_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'rand_normal'
    tumor_rand_tiles_dir = BASE_DIR / 'results'/ 'images' / 'selected_tiles' / 'rand_tumor'

    splits_dir = BASE_DIR / 'assets' / 'data_splits'

    if not os.path.exists(normal_images):
        sys.stderr.write(f"File \"{normal_images}\" not found")
        exit(1)

    if not os.path.exists(tumor_images):
        sys.stderr.write(f"File \"{tumor_images}\" not found")
        exit(1)

    if not os.path.exists(splits_dir):
        sys.stderr.write(f"File \"{splits_dir}\" not found")
        exit(1)

    if not path.exists(results):
        os.mkdir(results)

    if not os.path.exists(BASE_DIR / 'results' / 'images'):
        os.mkdir(BASE_DIR / 'results' / 'images')

    if not os.path.exists(BASE_DIR / 'results' / 'images' / 'selected_tiles'):
        os.mkdir(BASE_DIR / 'results' / 'images' / 'selected_tiles')

    if not os.path.exists(BASE_DIR / 'results' / 'images' / 'masked_images'):
        os.mkdir(BASE_DIR / 'results' / 'images' / 'masked_images')

    if not os.path.exists(BASE_DIR / 'results'/ 'images' / 'low_res_images'):
        os.mkdir(BASE_DIR / 'results' / 'images' / 'low_res_images')

    if not os.path.exists(normal_selected_tiles_dir):
        os.mkdir(normal_selected_tiles_dir)

    if not os.path.exists(tumor_selected_tiles_dir):
        os.mkdir(tumor_selected_tiles_dir)

    if not os.path.exists(selected_tiles_dir):
        os.mkdir(selected_tiles_dir)

    if not os.path.exists(normal_masked_images_dir):
        os.mkdir(normal_masked_images_dir)

    if not os.path.exists(tumor_masked_images_dir):
        os.mkdir(tumor_masked_images_dir)

    if not os.path.exists(low_res_normal_images_dir):
        os.mkdir(low_res_normal_images_dir)

    if not os.path.exists(low_res_tumor_images_dir):
        os.mkdir(low_res_tumor_images_dir)

    if not os.path.exists(normal_rand_tiles_dir):
        os.mkdir(normal_rand_tiles_dir)

    if not os.path.exists(tumor_rand_tiles_dir):
        os.mkdir(tumor_rand_tiles_dir)

    # Exploratory data analysis
    # TODO

    # Read slides info
    print("Normal slides info:")
    normal_slides_info = slide_info.read_slides_info(normal_images)
    slide_info.save_slides_info(normal_slides_info, results, "normal_slides_info.txt", display_info=False)

    print("\nTumor slides info:")
    tumor_slides_info = slide_info.read_slides_info(tumor_images)
    slide_info.save_slides_info(tumor_slides_info, results, "tumor_slides_info.txt", display_info=False)

    desired_magnification = 10
    tile_size = 224
    scale_factor = 32
    # Images preprocessing

    print("\nNormal images preprocessing:")
    preprocessing.preprocessing_images(normal_slides_info, selected_tiles_dir,
                                       os.path.join(results, "normal_filter_info.txt"),
                                       scale_factor, tile_size, desired_magnification,
                                       low_res_normal_images_dir, normal_masked_images_dir)

    print("\nTumor images preprocessing:")

    preprocessing.preprocessing_images(tumor_slides_info, selected_tiles_dir,
                                       os.path.join(results, "tumor_filter_info.txt"),
                                       scale_factor, tile_size, desired_magnification,
                                       low_res_tumor_images_dir, tumor_masked_images_dir)

    normal_slides_info.extend(tumor_slides_info)
    train_slides_info, test_slides_info, y_train, y_test = split_data.get_images_split_data(normal_slides_info, splits_dir)

    # features extraction
    print("\nImages feature extraction:")
    if args.method == 'fine_tuning':
        print(">> Fine tuning:")
        # TODO
        extracted_features_train_dir = Path('results') / 'images' / 'fine_tuning' / 'extracted_features' / 'numpy_train'
        extracted_features_test_dir = Path('results') / 'images' / 'fine_tuning' / 'extracted_features' / 'numpy_test'

        if not os.path.exists(Path('results') / 'images' / 'fine_tuning'):
            os.mkdir(Path('results') / 'images' / 'fine_tuning')

        if not os.path.exists(Path('results') / 'images' / 'fine_tuning' / 'extracted_features'):
            os.mkdir(Path('results') / 'images' / 'fine_tuning' / 'extracted_features')

        if not os.path.exists(extracted_features_train_dir):
            os.mkdir(extracted_features_train_dir)

        if not os.path.exists(extracted_features_test_dir):
            os.mkdir(extracted_features_test_dir)

        fine_tuning(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir)

    elif args.method == 'fixed_feature_generator':
        print(">> Fixed feature generator:")

        extracted_features_train_dir = Path('results') / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'numpy_train'
        extracted_features_test_dir = Path('results') / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'numpy_test'

        if not os.path.exists(Path('results') / 'images' / 'fixed_feature_generator'):
            os.mkdir(Path('results') / 'images' / 'fixed_feature_generator')

        if not os.path.exists(Path('results') / 'images' / 'fixed_feature_generator' / 'extracted_features'):
            os.mkdir(Path('results') / 'images' / 'fixed_feature_generator' / 'extracted_features')

        if not os.path.exists(extracted_features_train_dir):
            os.mkdir(extracted_features_train_dir)

        if not os.path.exists(extracted_features_test_dir):
            os.mkdir(extracted_features_test_dir)

        print(">> Extracting features from training images:")
        fixed_feature_generator(train_slides_info, extracted_features_train_dir, selected_tiles_dir,
                                tile_size, desired_magnification, USE_GPU)

        print(">> Extracting features from test images:")
        fixed_feature_generator(test_slides_info, extracted_features_test_dir, selected_tiles_dir,
                                tile_size, desired_magnification, USE_GPU)

    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == '__main__':
    main()
