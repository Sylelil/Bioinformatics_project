import sys
from openslide.deepzoom import DeepZoomGenerator
from tqdm import tqdm
import os
from config import paths
from . import utils


def read_slides_info_from_folder(lookup_dir):
    """
        Description: read slides info from folder, returning list of slides info and list of labels
        :param lookup_dir: path to directory
        :return: X, y: list of data and list of labels
    """
    X = []
    y = []

    for file in tqdm(os.listdir(lookup_dir), desc=">> Reading slides info...", file=sys.stdout):
        file_path = os.path.join(lookup_dir, file)
        slide = utils.open_wsi(file_path)
        slide_magnification = utils.get_wsi_magnification(slide)

        zoom = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=True)
        highest_zoom_level = utils.get_wsi_highest_zoom_level(zoom)

        (width, height) = slide.dimensions  # slide dimensions for level 0 (highest resolution)

        file_name = utils.extract_file_name_from_string_path(file_path)
        slide_info_dict = {
            'slide_path': file_path,
            'slide_name': file_name,  # slide name without extension
            'num_zoom_levels': zoom.level_count,
            'highest_zoom_level': highest_zoom_level,
            'slide_magnification': slide_magnification,
            'slide_width': width,
            'slide_height': height,
            'label': 0 if file_name.endswith("0") else 1,
        }
        X.append(slide_info_dict)
        y.append(slide_info_dict['label'])

    return X, y


def read_slides_info():
    """
        Description: read directory containing Whole slide images
        :return: list of dictionaries containing slides info
    """
    normal_slides_info = []
    tumor_slides_info = []

    for file in tqdm(os.listdir(paths.images_dir), desc=">> Reading slides info...", file=sys.stdout):
        current_dir = paths.images_dir
        if file.endswith('.svs'):
            slide = utils.open_wsi(os.path.join(current_dir, file))
            slide_magnification = utils.get_wsi_magnification(slide)

            zoom = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=True)
            highest_zoom_level = utils.get_wsi_highest_zoom_level(zoom)

            (width, height) = slide.dimensions  # slide dimensions for level 0 (highest resolution)

            file_name = utils.extract_file_name_from_string_path(os.path.join(current_dir, file))
            slide_info_dict = {
                'slide_path': os.path.join(current_dir, file),
                'slide_name': file_name,  # slide name without extension
                'num_zoom_levels': zoom.level_count,
                'highest_zoom_level': highest_zoom_level,
                'slide_magnification': slide_magnification,
                'slide_width': width,
                'slide_height': height,
                'label': 0 if file_name.endswith("0") else 1,
            }
            if slide_info_dict['label'] == 0:
                normal_slides_info.append(slide_info_dict)
            else:
                tumor_slides_info.append(slide_info_dict)

    return normal_slides_info, tumor_slides_info


def save_slides_info(slides_info, file_name, display_info=False):
    """
        Description: save Whole slide images info on disk

        :param slides_info: list of dictionaries containing slides info
        :param file_name: file name to save slide info
        :param display_info: Boolean
            if True shows slide info on screen
    """
    if display_info:
        for item in slides_info:
            print("%s: slide width= %d, slide height = %d, num of zoom levels = %d,"
                  " highest zoom level = %d, slide magn = %dx"
                  % (item['slide_name'], item['slide_width'],
                     item['slide_height'], item['num_zoom_levels'],
                     item['highest_zoom_level'], item['slide_magnification']))

    images_info_string = "slide number,slide name,slide weight,slide height,number of zoom " \
                         "levels,highest zoom level,slide magnification"
    for i in tqdm(range(0, len(slides_info)), desc=">> Saving slides info into file...", file=sys.stdout):
        images_info_string += "\n%d,%s,%d,%d,%d,%d,%d" % (i + 1, slides_info[i]['slide_name'],
                                                          slides_info[i]['slide_width'],
                                                          slides_info[i]['slide_height'],
                                                          slides_info[i]['num_zoom_levels'],
                                                          slides_info[i]['highest_zoom_level'],
                                                          slides_info[i]['slide_magnification'])
    images_info_string += "\n"

    if not os.path.exists(paths.images_results):
        os.mkdir(paths.images_results)

    images_info_file = open(os.path.join(paths.images_results, file_name), "w")
    images_info_file.write(images_info_string)
    images_info_file.close()
    print(">> Slides info saved to \"%s\"" % os.path.join(paths.images_results, file_name))
