import argparse
import os
import sys
from os import path
from pathlib import Path

import config.images.config as cfg
from config.images.config import BASE_DIR
from images import preprocessing, slide_info, utils
from images.features_extraction_methods.fine_tuning import fine_tuning
from images.features_extraction_methods.fixed_feature_generator import fixed_feature_generator
from common import split_data


USE_GPU = True


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
    images = BASE_DIR / 'datasets' / 'images'
    selected_coords_dir = cfg.selected_coords_dir

    normal_masked_images_dir = BASE_DIR / 'results' / 'images' / 'masked_images' / 'img_normal'
    tumor_masked_images_dir = BASE_DIR / 'results' / 'images' / 'masked_images' / 'img_tumor'

    low_res_normal_images_dir = BASE_DIR / 'results' / 'images' / 'low_res_images' / 'img_normal'
    low_res_tumor_images_dir = BASE_DIR / 'results' / 'images' / 'low_res_images' / 'img_tumor'

    normal_rand_tiles_dir = BASE_DIR / 'results' / 'images' / 'selected_tiles' / 'rand_normal'
    tumor_rand_tiles_dir = BASE_DIR / 'results'/ 'images' / 'selected_tiles' / 'rand_tumor'

    splits_dir = BASE_DIR / 'assets' / 'data_splits'

    if not os.path.exists(images):
        sys.stderr.write(f"File \"{images}\" not found")
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

    if not os.path.exists(selected_coords_dir):
        os.mkdir(selected_coords_dir)

    if not os.path.exists(cfg.selected_tiles_dir):
        os.mkdir(cfg.selected_tiles_dir)

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
    print("Slides info:")
    normal_slides_info, tumor_slides_info = slide_info.read_slides_info(images)
    slide_info.save_slides_info(normal_slides_info, results, "normal_slides_info.txt", display_info=False)
    slide_info.save_slides_info(tumor_slides_info, results, "tumor_slides_info.txt", display_info=False)

    # Images preprocessing
    print("\nNormal images preprocessing:")
    preprocessing.preprocessing_images(normal_slides_info, selected_coords_dir,
                                       os.path.join(results, "normal_filter_info.txt"),
                                       os.path.join(results, "normal_tiles_info.txt"),
                                       low_res_normal_images_dir, normal_masked_images_dir)

    print("\nTumor images preprocessing:")
    preprocessing.preprocessing_images(tumor_slides_info, selected_coords_dir,
                                       os.path.join(results, "tumor_filter_info.txt"),
                                       os.path.join(results, "normal_tiles_info.txt"),
                                       low_res_tumor_images_dir, tumor_masked_images_dir)

    slides_info = normal_slides_info + tumor_slides_info
    print("\nSaving selected tiles on disk:")
    preprocessing.extract_tiles_on_disk(slides_info)

    print("\nSplitting data in train and test:")
    print(f'>> Tot data: {len(slides_info)}')
    train_slides_info, test_slides_info, y_train, y_test = split_data.get_images_split_data(slides_info, splits_dir) #TODO splits

    # Compute number of samples
    train_slides_info_0 = [slide for slide in train_slides_info if slide['label'] == 0]
    train_slides_info_1 = [slide for slide in train_slides_info if slide['label'] == 1]

    print(f'\nTraining data:\n>> Tot = {len(train_slides_info)}\n'
          f'>> Tumor samples = {len(train_slides_info_1)}\n>> Normal samples = {len(train_slides_info_0)}')

    test_slides_info_0 = [slide for slide in test_slides_info if slide['label'] == 0]
    test_slides_info_1 = [slide for slide in test_slides_info if slide['label'] == 1]

    print(f'\nTest data:\n>> Tot = {len(test_slides_info)}\n'
          f'>> Tumor samples = {len(test_slides_info_1)}\n>> Normal samples = {len(test_slides_info_0)}')

    # features extraction
    print("\nImages feature extraction:")
    if args.method == 'fine_tuning':
        print(">> Fine tuning:")
        extracted_features_train_dir = BASE_DIR / 'results' / 'images' / 'fine_tuning' / 'extracted_features' / 'train'
        extracted_features_test_dir = BASE_DIR / 'results' / 'images' / 'fine_tuning' / 'extracted_features' / 'test'

        if not os.path.exists(BASE_DIR / 'results' / 'images' / 'fine_tuning'):
            os.mkdir(BASE_DIR / 'results' / 'images' / 'fine_tuning')

        if not os.path.exists(BASE_DIR / 'results' / 'images' / 'fine_tuning' / 'extracted_features'):
            os.mkdir(BASE_DIR / 'results' / 'images' / 'fine_tuning' / 'extracted_features')

        if not os.path.exists(extracted_features_train_dir):
            os.mkdir(extracted_features_train_dir)

        if not os.path.exists(extracted_features_test_dir):
            os.mkdir(extracted_features_test_dir)

        print(train_slides_info[0])
        fine_tuning(train_slides_info, test_slides_info, y_train, y_test)

    elif args.method == 'fixed_feature_generator':
        print(">> Fixed feature generator:")

        extracted_features_train_dir = Path('results') / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'train'
        extracted_features_test_dir = Path('results') / 'images' / 'fixed_feature_generator' / 'extracted_features' / 'test'

        if not os.path.exists(Path('results') / 'images' / 'fixed_feature_generator'):
            os.mkdir(Path('results') / 'images' / 'fixed_feature_generator')

        if not os.path.exists(Path('results') / 'images' / 'fixed_feature_generator' / 'extracted_features'):
            os.mkdir(Path('results') / 'images' / 'fixed_feature_generator' / 'extracted_features')

        if not os.path.exists(extracted_features_train_dir):
            os.mkdir(extracted_features_train_dir)

        if not os.path.exists(extracted_features_test_dir):
            os.mkdir(extracted_features_test_dir)

        print(">> Extracting features from training images:")
        fixed_feature_generator(train_slides_info, extracted_features_train_dir, selected_coords_dir,
                                USE_GPU)

        print(">> Extracting features from test images:")
        fixed_feature_generator(test_slides_info, extracted_features_test_dir, selected_coords_dir,
                                USE_GPU)

    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    main()


