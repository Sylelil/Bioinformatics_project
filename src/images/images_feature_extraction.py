import argparse
import os
import sys
from os import path
from pathlib import Path
from src.images import preprocessing, slide_info, utils
from src.images.features_extraction_methods.fine_tuning import fine_tuning
from src.images.features_extraction_methods.fixed_feature_generator import fixed_feature_generator

USE_GPU = False


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
    results = Path('results') / 'images'
    normal_images = Path('datasets') / 'images' / 'normal'
    tumor_images = Path('datasets') / 'images' / 'tumor'

    numpy_normal_dir = Path('results') / 'images' / 'extracted_features' / 'numpy_normal'
    numpy_tumor_dir = Path('results') / 'images' / 'extracted_features' / 'numpy_tumor'

    normal_selected_tiles_dir = Path('results') / 'images' / 'selected_tiles' / 'numpy_normal'
    tumor_selected_tiles_dir = Path('results') / 'images' / 'selected_tiles' / 'numpy_tumor'

    heatmap_dir = Path('results') / 'images' / 'masked_images' / 'heatmap'

    normal_masked_images_dir = Path('results') / 'images' / 'masked_images' / 'img_normal'
    tumor_masked_images_dir = Path('results') / 'images' / 'masked_images' / 'img_tumor'

    normal_images_dir = Path('results') / 'images' / 'low_res_images' / 'img_normal'
    tumor_images_dir = Path('results') / 'images' / 'low_res_images' / 'img_tumor'

    normal_rand_tiles_dir = Path('results') / 'images' / 'selected_tiles' / 'rand_normal'
    tumor_rand_tiles_dir = Path('results') / 'images' / 'selected_tiles' / 'rand_tumor'

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

    if not os.path.exists(Path('results') / 'images' / 'low_res_images'):
        os.mkdir(Path('results') / 'images' / 'low_res_images')

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

    if not os.path.exists(heatmap_dir):
        os.mkdir(heatmap_dir)

    if not os.path.exists(normal_images_dir):
        os.mkdir(normal_images_dir)

    if not os.path.exists(tumor_images_dir):
        os.mkdir(tumor_images_dir)

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
    '''
    slide = {}
    for s in tumor_slides_info:
        if s['slide_name'] == "86d18004-9478-4e46-83b4-4fbe445ccb70_1":
            slide = s
            print("ok")
            break

    scaled_image, scaled_w, scaled_h = utils.from_wsi_to_scaled_pillow_image(slide['slide_path'], scale_factor)
    print(scaled_h)
    print(scaled_w)
    preprocessing.apply_filters_to_image(slide, scaled_image, tumor_masked_images_dir)

    exit()
    '''
    print("\nNormal images preprocessing:")
    preprocessing.preprocessing_images(normal_slides_info, normal_selected_tiles_dir,
                                       os.path.join(results, "normal_filter_info.txt"),
                                       scale_factor, tile_size, desired_magnification,
                                       heatmap_dir, normal_images_dir, normal_masked_images_dir)

    print("\nTumor images preprocessing:")

    preprocessing.preprocessing_images(tumor_slides_info, tumor_selected_tiles_dir,
                                       os.path.join(results, "tumor_filter_info.txt"),
                                       scale_factor, tile_size, desired_magnification,
                                       heatmap_dir, tumor_images_dir, tumor_masked_images_dir)

    # features extraction
    print("\nImages feature extraction:")
    if args.method == 'fine_tuning':
        # TODO
        fine_tuning()
    elif args.method == 'fixed_feature_generator':
        print(">> Fixed feature generator:")
        print(">> Extracting features from normal images:")
        fixed_feature_generator(normal_slides_info, numpy_normal_dir, normal_selected_tiles_dir, normal_rand_tiles_dir,
                                tile_size, desired_magnification, USE_GPU)

        print(">> Extracting features from tumor images:")
        fixed_feature_generator(tumor_slides_info, numpy_tumor_dir, tumor_selected_tiles_dir, tumor_rand_tiles_dir,
                                tile_size, desired_magnification, USE_GPU)
    else:
        sys.stderr.write("Invalid value for <feature extraction method> in config file")
        exit(1)


if __name__ == '__main__':
    main()
