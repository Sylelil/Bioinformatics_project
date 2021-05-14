import argparse
import os
import sys
from os import path
from pathlib import Path
import config.images.config as cfg
from config import paths
from images import preprocessing, slide_info
from images.features_extraction_method.fine_tuning import fine_tuning
from images.features_extraction_method.fixed_feature_generator import fixed_feature_generator
from src.common import split_data


def main():
    if not cfg.USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Parse arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--method',
                        help='Feature extraction method',
                        choices=['fine_tuning', 'fixed_feature_generator'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    if not os.path.exists(paths.images_dir):
        sys.stderr.write(f"File \"{paths.images_dir}\" not found")
        exit(1)

    if not os.path.exists(paths.split_data_dir):
        sys.stderr.write(f"File \"{paths.split_data_dir}\" not found")
        exit(1)

    normal_masked_images_dir = Path(paths.images_results) / 'masked_images' / 'img_normal'
    tumor_masked_images_dir = Path(paths.images_results) / 'masked_images' / 'img_tumor'
    low_res_normal_images_dir = Path(paths.images_results) / 'low_res_images' / 'img_normal'
    low_res_tumor_images_dir = Path(paths.images_results) / 'low_res_images' / 'img_tumor'

    if not path.exists(paths.images_results):
        os.makedirs(paths.images_results)

    if not os.path.exists(paths.selected_coords_dir):
        os.makedirs(paths.selected_coords_dir)

    if not os.path.exists(paths.selected_tiles_dir):
        os.makedirs(paths.selected_tiles_dir)

    if not os.path.exists(normal_masked_images_dir):
        os.makedirs(normal_masked_images_dir)

    if not os.path.exists(tumor_masked_images_dir):
        os.makedirs(tumor_masked_images_dir)

    if not os.path.exists(low_res_normal_images_dir):
        os.makedirs(low_res_normal_images_dir)

    if not os.path.exists(low_res_tumor_images_dir):
        os.makedirs(low_res_tumor_images_dir)

    if not os.path.exists(paths.extracted_features_train):
        os.makedirs(paths.extracted_features_train)

    if not os.path.exists(paths.extracted_features_val):
        os.makedirs(paths.extracted_features_val)

    if not os.path.exists(paths.extracted_features_test):
        os.makedirs(paths.extracted_features_test)

    # Read slides info
    print("Slides info:")
    normal_slides_info, tumor_slides_info = slide_info.read_slides_info()
    slide_info.save_slides_info(normal_slides_info, "normal_slides_info.txt", display_info=False)
    slide_info.save_slides_info(tumor_slides_info, "tumor_slides_info.txt", display_info=False)

    # Images preprocessing
    print("\nNormal images preprocessing:")
    preprocessing.preprocessing_images(normal_slides_info, paths.selected_coords_dir,
                                       os.path.join(paths.images_results, "normal_filter_info.txt"),
                                       os.path.join(paths.images_results, "normal_tiles_info.txt"),
                                       low_res_normal_images_dir, normal_masked_images_dir)

    print("\nTumor images preprocessing:")
    preprocessing.preprocessing_images(tumor_slides_info, paths.selected_coords_dir,
                                       os.path.join(paths.images_results, "tumor_filter_info.txt"),
                                       os.path.join(paths.images_results, "tumor_tiles_info.txt"),
                                       low_res_tumor_images_dir, tumor_masked_images_dir)

    slides_info = normal_slides_info + tumor_slides_info

    rand_tiles_dir = paths.images_results / 'selected_tiles' / 'rand_tiles'
    low_res_rand_tiles_dir = paths.images_results / 'selected_tiles' / 'low_res_rand_tiles'
    if not os.path.exists(rand_tiles_dir):
        os.makedirs(rand_tiles_dir)

    if not os.path.exists(low_res_rand_tiles_dir):
        os.makedirs(low_res_rand_tiles_dir)

    for slide in slides_info:
        preprocessing.plot_random_selected_tiles(slide, rand_tiles_dir, num_tiles=16)

    if args.method == 'fine_tuning':
        print("\nSaving selected tiles on disk:")
        preprocessing.extract_tiles_on_disk(slides_info)

    print("\nReading split slides data:")
    images_splits_path = Path(paths.split_data_dir) / 'images'
    print(f'>> Tot data: {len(slides_info)}')
    if not os.path.exists(paths.split_data_dir):
        print("%s not existing." % paths.split_data_dir)
        exit()
    if not os.path.exists(images_splits_path):
        print("%s not existing." % images_splits_path)
        exit()

    train_slides_info, val_slides_info, test_slides_info, y_train, y_val, y_test = split_data.get_images_split_data(images_splits_path, val_data=True)

    print(train_slides_info)
    # Compute number of samples
    train_slides_info_0 = [slide for slide in train_slides_info if slide['label'] == 0]
    train_slides_info_1 = [slide for slide in train_slides_info if slide['label'] == 1]

    print(f'\nTraining data:\n>> Tot = {len(train_slides_info)}\n'
          f'>> Tumor samples = {len(train_slides_info_1)}\n>> Normal samples = {len(train_slides_info_0)}')

    val_slides_info_0 = [slide for slide in val_slides_info if slide['label'] == 0]
    val_slides_info_1 = [slide for slide in val_slides_info if slide['label'] == 1]

    print(f'\nVal data:\n>> Tot = {len(val_slides_info)}\n'
          f'>> Tumor samples = {len(val_slides_info_1)}\n>> Normal samples = {len(val_slides_info_0)}')

    test_slides_info_0 = [slide for slide in test_slides_info if slide['label'] == 0]
    test_slides_info_1 = [slide for slide in test_slides_info if slide['label'] == 1]

    print(f'\nTest data:\n>> Tot = {len(test_slides_info)}\n'
          f'>> Tumor samples = {len(test_slides_info_1)}\n>> Normal samples = {len(test_slides_info_0)}')

    # features extraction
    print("\nImages feature extraction:")
    if args.method == 'fine_tuning':
        print(">> Fine tuning:")

        print(train_slides_info[0])
        fine_tuning(train_slides_info, test_slides_info, y_train, y_test)

    elif args.method == 'fixed_feature_generator':
        print(">> Fixed feature generator:")

        print(">> Extracting features from training images:")
        fixed_feature_generator(train_slides_info, paths.extracted_features_train, paths.selected_coords_dir)

        print(">> Extracting features from val images:")
        fixed_feature_generator(val_slides_info, paths.extracted_features_val, paths.selected_coords_dir)

        print(">> Extracting features from test images:")
        fixed_feature_generator(test_slides_info, paths.extracted_features_test, paths.selected_coords_dir)

    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    main()


