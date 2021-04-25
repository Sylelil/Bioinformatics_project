import argparse
import os
import sys
from os import path
from pathlib import Path
import config.images.config as cfg
from images import preprocessing, slide_info
from images.features_extraction_methods.fine_tuning import fine_tuning
from images.features_extraction_methods.fixed_feature_generator import fixed_feature_generator
from common import split_data


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

    if not os.path.exists(cfg.images):
        sys.stderr.write(f"File \"{cfg.images}\" not found")
        exit(1)

    if not os.path.exists(cfg.splits_dir):
        sys.stderr.write(f"File \"{cfg.splits_dir}\" not found")
        exit(1)

    if not path.exists(cfg.results):
        os.makedirs(cfg.results)

    if not os.path.exists(cfg.selected_coords_dir):
        os.makedirs(cfg.selected_coords_dir)

    if not os.path.exists(cfg.selected_tiles_dir):
        os.makedirs(cfg.selected_tiles_dir)

    if not os.path.exists(cfg.normal_masked_images_dir):
        os.makedirs(cfg.normal_masked_images_dir)

    if not os.path.exists(cfg.tumor_masked_images_dir):
        os.makedirs(cfg.tumor_masked_images_dir)

    if not os.path.exists(cfg.low_res_normal_images_dir):
        os.makedirs(cfg.low_res_normal_images_dir)

    if not os.path.exists(cfg.low_res_tumor_images_dir):
        os.makedirs(cfg.low_res_tumor_images_dir)

    if not os.path.exists(cfg.normal_rand_tiles_dir):
        os.makedirs(cfg.normal_rand_tiles_dir)

    if not os.path.exists(cfg.tumor_rand_tiles_dir):
        os.makedirs(cfg.tumor_rand_tiles_dir)

    if not os.path.exists(cfg.extracted_features_train_dir):
        os.makedirs(cfg.extracted_features_train_dir)

    if not os.path.exists(cfg.extracted_features_test_dir):
        os.makedirs(cfg.extracted_features_test_dir)

    # Read slides info
    print("Slides info:")
    normal_slides_info, tumor_slides_info = slide_info.read_slides_info()
    slide_info.save_slides_info(normal_slides_info, "normal_slides_info.txt", display_info=False)
    slide_info.save_slides_info(tumor_slides_info, "tumor_slides_info.txt", display_info=False)

    # Images preprocessing
    print("\nNormal images preprocessing:")
    preprocessing.preprocessing_images(normal_slides_info, cfg.selected_coords_dir,
                                       os.path.join(cfg.results, "normal_filter_info.txt"),
                                       os.path.join(cfg.results, "normal_tiles_info.txt"),
                                       cfg.low_res_normal_images_dir, cfg.normal_masked_images_dir)

    print("\nTumor images preprocessing:")
    preprocessing.preprocessing_images(tumor_slides_info, cfg.selected_coords_dir,
                                       os.path.join(cfg.results, "tumor_filter_info.txt"),
                                       os.path.join(cfg.results, "normal_tiles_info.txt"),
                                       cfg.low_res_tumor_images_dir, cfg.tumor_masked_images_dir)

    slides_info = normal_slides_info + tumor_slides_info
    print("\nSaving selected tiles on disk:")
    preprocessing.extract_tiles_on_disk(slides_info)

    print("\nSplitting data in train and test:")
    print(f'>> Tot data: {len(slides_info)}')
    train_slides_info, test_slides_info, y_train, y_test = split_data.get_images_split_data(slides_info, cfg.splits_dir) #TODO splits

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

        print(train_slides_info[0])
        fine_tuning(train_slides_info, test_slides_info, y_train, y_test)

    elif args.method == 'fixed_feature_generator':
        print(">> Fixed feature generator:")

        print(">> Extracting features from training images:")
        fixed_feature_generator(train_slides_info, cfg.extracted_features_train_dir, cfg.selected_coords_dir)

        print(">> Extracting features from test images:")
        fixed_feature_generator(test_slides_info, cfg.extracted_features_test_dir, cfg.selected_coords_dir)

    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == '__main__':
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    main()


