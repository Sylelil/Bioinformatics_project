import argparse
import os
import sys
from os import path
from pathlib import Path
from src.images import utils
from src.images.features_extraction_methods.fine_tuning import fine_tuning
from src.images.features_extraction_methods.fixed_feature_generator import fixed_feature_generator


def main():
    # Parse arguments from command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--method',
                        help='Feature extraction method',
                        choices=['fine_tuning', 'fixed_feature_generator'],
                        required=True,
                        type=str)

    args = parser.parse_args()

    # Paths
    results = Path('results') / 'images'
    normal_images = Path('datasets') / 'images' / 'normal'
    tumor_images = Path('datasets') / 'images' / 'tumor'

    numpy_normal_dir = Path('results') / 'images' / 'extracted_features' / 'numpy_normal'
    numpy_tumor_dir = Path('results') / 'images' / 'extracted_features' / 'numpy_tumor'

    normal_selected_tiles_dir = Path('results') / 'images' / 'selected_tiles' / 'numpy_normal'
    tumor_selected_tiles_dir = Path('results') / 'images' / 'selected_tiles' / 'numpy_tumor'

    normal_masked_images_dir = Path('results') / 'images' / 'masked_images' / 'numpy_normal'
    tumor_masked_images_dir = Path('results') / 'images' / 'masked_images' / 'numpy_tumor'

    if not os.path.exists(normal_images):
        sys.stderr.write(f"File \"{normal_images}\" not found")
        exit(1)

    if not os.path.exists(tumor_images):
        sys.stderr.write(f"File \"{tumor_images}\" not found")
        exit(1)

    if not path.exists(results):
        os.mkdir(results)

    if not os.path.exists(Path('results') / 'images'):
        os.mkdir(Path('results') / 'images')

    if not os.path.exists(Path('results') / 'images' / 'extracted_features'):
        os.mkdir(Path('results') / 'images' / 'extracted_features')

    if not os.path.exists(Path('results') / 'images' / 'selected_tiles'):
        os.mkdir(Path('results') / 'images' / 'selected_tiles')

    if not os.path.exists(Path('results') / 'images' / 'masked_images'):
        os.mkdir(Path('results') / 'images' / 'masked_images')

    if not os.path.exists(normal_selected_tiles_dir):
        os.mkdir(normal_selected_tiles_dir)

    if not os.path.exists(tumor_selected_tiles_dir):
        os.mkdir(tumor_selected_tiles_dir)

    if not os.path.exists(normal_masked_images_dir):
        os.mkdir(normal_masked_images_dir)

    if not os.path.exists(tumor_masked_images_dir):
        os.mkdir(tumor_masked_images_dir)

    if not os.path.exists(numpy_normal_dir):
        os.mkdir(numpy_normal_dir)

    if not os.path.exists(numpy_tumor_dir):
        os.mkdir(numpy_tumor_dir)

    # Exploratory data analysis
    #TODO

    # Read slides info
    print("Normal slides info:")
    normal_slides_info = utils.read_slides_info(normal_images)
    utils.save_slides_info(normal_slides_info, results, "normal_slides_info.txt", display_info=False)

    print("\nTumor slides info:")
    tumor_slides_info = utils.read_slides_info(tumor_images)
    utils.save_slides_info(tumor_slides_info, results, "tumor_slides_info.txt", display_info=False)

    desired_magnification = 10
    tile_size = 224
    scale_factor = 32
    # Images preprocessing
    print("\nNormal images preprocessing:")
    normal_slides_tiles_coords = utils.preprocessing_images(normal_slides_info, normal_selected_tiles_dir,
                                                            os.path.join(results, "normal_filter_info.txt"),
                                                            scale_factor, tile_size, desired_magnification, normal_masked_images_dir)

    print("\nTumor images preprocessing:")
    tumor_slides_tiles_coords = utils.preprocessing_images(tumor_slides_info, tumor_selected_tiles_dir,
                                                           os.path.join(results, "tumor_filter_info.txt"),
                                                           scale_factor, tile_size, desired_magnification, tumor_masked_images_dir)
    # features extraction
    print("\nImages feature extraction:")
    if args.method == 'fine_tuning':
        # TODO
        fine_tuning()
    elif args.method == 'fixed_feature_generator':
        print("\nFixed feature generator:")
        print("Extracting features from normal images:")
        fixed_feature_generator(normal_slides_tiles_coords, normal_slides_info, numpy_normal_dir,
                                tile_size, desired_magnification)

        print("\nExtracting features from tumor images:")
        fixed_feature_generator(tumor_slides_tiles_coords, tumor_slides_info, numpy_tumor_dir,
                                tile_size, desired_magnification)
    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == '__main__':
    main()
