import os
from enum import Enum
import openslide
from openslide.deepzoom import DeepZoomGenerator
from openslide import OpenSlideError
from tqdm import tqdm
import datetime
import multiprocessing
import math
import sys
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import skimage.morphology as sk_morphology
import skimage.filters as sk_filters
from skimage import img_as_bool
from skimage.filters import median, gaussian
import skimage.color as sk_color
from skimage.morphology import disk
from os import path


class Time:
    """
    Class for displaying elapsed time.
    """

    def __init__(self):
        self.start = datetime.datetime.now()

    def elapsed_display(self):
        time_elapsed = self.elapsed()
        print("Time elapsed: " + str(time_elapsed))

    def elapsed(self):
        self.end = datetime.datetime.now()
        time_elapsed = self.end - self.start
        return time_elapsed


def open_wsi(file_path):
    """
       Description: open Whole Slide Image
       :param file_path: path to wsi
       :return: OpenSlide object
    """
    try:
        slide = openslide.open_slide(file_path)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide


def get_wsi_magnification(slide):
    """
        :param slide: OpenSlide object
        :return: wsi magnification
    """
    return int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])


def get_wsi_highest_zoom_level(generator):
    """
        :param generator: DeepZoomGenerator that wraps an OpenSlide object
        :return: highest wsi level
      """
    return generator.level_count - 1  # 0-based indexing


def extract_file_name_from_string_path(file_path):
    """
        :param file_path: path to file
        :return: file name
    """
    base = os.path.basename(file_path)
    return os.path.splitext(base)[0]


def read_slides_info(lookup_dir):
    """
        Description: read directory containing Whole slide images
        :param lookup_dir: path to directory
        :return: list of dictionaries containing slides info, #TODO
    """
    slides_info = []

    for _dir in tqdm(os.listdir(lookup_dir), desc=">> Reading slides info...", file=sys.stdout):
        current_dir = os.path.join(lookup_dir, _dir)
        if os.path.isdir(current_dir):
            for file in os.listdir(current_dir):
                if file.endswith('.svs'):
                    slide = open_wsi(os.path.join(current_dir, file))
                    slide_magnification = get_wsi_magnification(slide)

                    zoom = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=True)
                    highest_zoom_level = get_wsi_highest_zoom_level(zoom)

                    (width, height) = slide.dimensions  # slide dimensions for level 0 (highest resolution)

                    file_name = extract_file_name_from_string_path(os.path.join(current_dir, file))
                    slide_info_dict = {
                        'slide_path': os.path.join(current_dir, file),
                        'slide_name': file_name,  # slide name without extension
                        'num_zoom_levels': zoom.level_count,
                        'highest_zoom_level': highest_zoom_level,
                        'slide_magnification': slide_magnification,
                        'slide_width': width,
                        'slide_height': height,
                    }
                    slides_info.append(slide_info_dict)

    return slides_info


def pil_to_np_rgb(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    Args:
      pil_img: The PIL Image.
    Returns:
      The PIL image converted to a NumPy array.
      :param pil_img:
      :param display_info:
    """
    t = Time()
    rgb = np.asarray(pil_img)
    return rgb


def apply_filters(start_ind, end_ind, slide_images, scale_factor):
    string = ""
    segmented_images = {}
    for slide_num in range(start_ind - 1, end_ind):
        scaled_image, scaled_w, scaled_h = from_wsi_to_scaled_pillow_image(slide_images[slide_num]['slide_path'], scale_factor)
        info, segmented_image = apply_filters_to_image(slide_images[slide_num], scaled_image)
        string += info + '\n'
        segmented_images[slide_images[slide_num]['slide_name']] = segmented_image
    return start_ind, end_ind, string, segmented_images


def apply_filters_to_image(slide_info, scaled_image):
    """
    Apply a set of filters to an image and optionally save and/or display filtered images.
    Args:
      slide_num: The slide number.
      save: If True, save filtered images.
      display: If True, display filtered images to screen.
    Returns:
      Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
      (used for HTML page generation).
    """
    string = "Slide %s:\n%-20s | Width: %d Height: %d\n" % (
        slide_info['slide_name'], "SVS", slide_info['slide_width'], slide_info['slide_height'])

    rgb = pil_to_np_rgb(scaled_image)
    string += np_info(rgb, "RGB", Time().elapsed())
    string += '\n'
    # utils.display_img(rgb, "RGB")

    # from RGB to grayscale
    grayscale = filter_rgb_to_grayscale(rgb)
    string += np_info(grayscale, "Gray", Time().elapsed())
    string += '\n'
    # utils.display_img(grayscale, "Grayscale")

    # otsu's adaptive thresholding
    # complement -> in order to have background values close to 0
    complement = filter_complement(grayscale)
    string += np_info(complement, "Complement", Time().elapsed())
    string += '\n'
    # utils.display_img(complement, "Complement")

    otsu_mask = filter_otsu_threshold(complement)
    string += np_info(otsu_mask, "Otsu Threshold", Time().elapsed())
    string += '\n'

    # utils.display_img(otsu_mask, "Compl. Otsu mask")

    median_filtering_otsu_mask = median(otsu_mask,
                                        disk(2))  # apply median filtering (radius = 2) on otsu mask (noise reduction)
    string += np_info(median_filtering_otsu_mask, "MEDIAN FILTERING")
    string += '\n'
    # utils.display_img(median_filtering_otsu_mask, "Compl. Otsu mask(median)")

    blurring_otsu_mask = gaussian(median_filtering_otsu_mask, sigma=2)
    string += np_info(blurring_otsu_mask, "BLURRING")
    string += '\n'
    # utils.display_img(blurring_otsu_mask, "Compl. Otsu mask(gaussian)")
    blurring_otsu_mask = img_as_bool(blurring_otsu_mask, force_copy=False)

    no_small_obj_otsu_mask = filter_remove_small_objects(blurring_otsu_mask, min_size=5000)
    string += np_info(no_small_obj_otsu_mask, "Remove Small Objs", Time().elapsed())
    string += '\n'
    # utils.display_img(no_small_obj_otsu_mask, "Compl. Otsu mask(obj.)")
    no_small_obj_otsu_mask = img_as_bool(no_small_obj_otsu_mask, force_copy=False)

    no_small_holes_otsu_mask = filter_remove_small_holes(no_small_obj_otsu_mask, max_size=100)
    string += np_info(no_small_holes_otsu_mask, "Remove Small Holes", Time().elapsed())
    string += '\n'
    # utils.display_img(no_small_holes_otsu_mask, "Compl. Otsu mask(holes)")

    no_small_holes_otsu_mask = img_as_bool(no_small_holes_otsu_mask, force_copy=False)
    segmented_image = mask_rgb(rgb,
                               no_small_holes_otsu_mask)  # pixel wise and between the original image and the complementary of the otsu mask
    string += np_info(no_small_holes_otsu_mask, "Mask RGB", Time().elapsed())
    string += '\n'

    # utils.display_img(segmented_image, "Segmented image", bg=True)
    pil_segmented_image = np_to_pil(segmented_image)

    print("Image " + slide_info['slide_name'] + " masked")

    return string, pil_segmented_image


def from_wsi_to_scaled_pillow_image(file_path, scale_factor):
    """
       Description : Convert a WSI training slide to a scaled-down PIL image.
       :param scale_factor:
       :param file_path:
       :return: Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height
    """
    # print("Opening Slide: %s" % file_path)
    slide = open_wsi(file_path)
    large_w, large_h = slide.dimensions

    new_w = math.floor(large_w / scale_factor)
    new_h = math.floor(large_h / scale_factor)

    level = slide.get_best_level_for_downsample(scale_factor)

    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")

    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, new_w, new_h


def multiprocess_apply_filters_to_wsi(slides_images, dest_file_path, scale_factor):
    """
    Convert all WSI training slides to smaller images using multiple processes (one process per core).
    Each process will process a range of slide numbers.
    """
    timer = Time()

    # how many processes to use
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    num_train_images = len(slides_images)
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes

    print("Number of processes: " + str(num_processes))
    print("Number of training images: " + str(num_train_images))

    # each task specifies a range of slides
    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        tasks.append((start_index, end_index, slides_images, scale_factor))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(apply_filters, t))

    filter_info = ""
    segmented_images = {}
    for result in results:
        (start_ind, end_ind, filter_info_range, segmented_images_range) = result.get()
        filter_info += filter_info_range
        segmented_images.update(segmented_images_range)
        if start_ind == end_ind:
            print("Done converting slide %d" % start_ind)
        else:
            print("Done converting slides %d through %d" % (start_ind, end_ind))

    print(">> Time to apply filters to all images (multiprocess): %s" % str(timer.elapsed()))

    images_info_file = open(dest_file_path, "w")
    images_info_file.write(filter_info)
    images_info_file.close()
    print(">> Filter info saved to \"%s\"\n" % dest_file_path)

    return segmented_images


def np_info(np_arr, name=None, elapsed=None):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.
    Args:
      np_arr: The NumPy array.
      name: The (optional) name of the array.
      elapsed: The (optional) time elapsed to perform a filtering operation.
    """

    if name is None:
        name = "NumPy Array"
    if elapsed is None:
        elapsed = "---"

    # np_arr = np.asarray(np_arr)
    max = np_arr.max()
    min = np_arr.min()
    mean = np_arr.mean()
    is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
    return "%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
            name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape)


def filter_rgb_to_grayscale(np_img, output_type="uint8"):
    """
  Convert an RGB NumPy array to a grayscale NumPy array.
  Shape (h, w, c) to (h, w).
  Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)
  Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    if output_type != "float":
        grayscale = grayscale.astype("uint8")
    return grayscale


def filter_otsu_threshold(np_img, output_type="uint8"):
    """
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
  Args:
    np_img: Image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    otsu = (np_img > otsu_thresh_value)
    '''
    if output_type == "bool":
        pass
    elif output_type == "float":
        otsu = otsu.astype(float)
    else:
        otsu = otsu.astype("uint8") * 255
        '''
    return otsu


def mask_rgb(rgb, mask):
    """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.
  Returns:
    NumPy array representing an RGB image with mask applied.
  """
    result = rgb * np.dstack([mask, mask, mask])
    return result


def filter_complement(np_img, output_type="uint8"):
    """
  Obtain the complement of an image as a NumPy array.
  Args:
    np_img: Image as a NumPy array.
    type: Type of array to return (float or uint8).
  Returns:
    Complement image as Numpy array.
  """
    if output_type == "float":
        complement = 1.0 - np_img
    else:
        complement = 255 - np_img
    return complement


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
    is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
    reduce the amount of masking that this filter performs.
    Args:
      np_img: Image as a NumPy array of type bool.
      min_size: Minimum size of small object to remove.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array (bool, float, or uint8).
    """

    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
        new_min_size = min_size // 2
        print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
            mask_percentage, overmask_thresh, min_size, new_min_size))
        rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    return np_img


def filter_remove_small_holes(np_img, max_size=3000, output_type="uint8"):
    """
    Filter image to remove small holes less than a particular size.
    Args:
      np_img: Image as a NumPy array of type bool.
      min_size: Remove small holes below this size.
      output_type: Type of array to return (bool, float, or uint8).
    Returns:
      NumPy array (bool, float, or uint8).
    """

    rem_sm = sk_morphology.remove_small_holes(np_img, area_threshold=max_size)

    if output_type == "bool":
        pass
    elif output_type == "float":
        rem_sm = rem_sm.astype(float)
    else:
        rem_sm = rem_sm.astype("uint8") * 255

    return rem_sm


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    Args:
      np_img: The image represented as a NumPy array.
    Returns:
       The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
  Args:
    np_img: Image as a NumPy array.
  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def save_slides_info(slides_info, save_dir, file_name, display_info=False):
    if display_info:
        for item in slides_info:
            print("%s: slide width= %d, slide height = %d, num of zoom levels = %d,"
                  " highest zoom level = %d, slide magn = %dx"
                  % (item['slide_name'], item['slide_width'],
                     item['slide_height'], item['num_zoom_levels'],
                     item['highest_zoom_level'], item['slide_magnification']))

    images_info_string = "slide number,slide name,slide weight,slide height,number of zoom levels,highest zoom level,slide magnification"
    for i in tqdm(range(0, len(slides_info)), desc=">> Saving slides info into file...", file=sys.stdout):
        images_info_string += "\n%d,%s,%d,%d,%d,%d,%d" % (i + 1, slides_info[i]['slide_name'],
                                                          slides_info[i]['slide_width'],
                                                          slides_info[i]['slide_height'],
                                                          slides_info[i]['num_zoom_levels'],
                                                          slides_info[i]['highest_zoom_level'],
                                                          slides_info[i]['slide_magnification'])
    images_info_string += "\n"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    images_info_file = open(os.path.join(save_dir, file_name), "w")
    images_info_file.write(images_info_string)
    images_info_file.close()
    print(">> Slides info saved to \"%s\"" % os.path.join(save_dir, file_name))


def multiprocess_select_tiles_with_tissue(slides_images, dict_masked_pil_images, selected_tiles_dir,
                                          tile_size, desired_magnification, scale_factor):
    """
    Convert all WSI training slides to smaller images using multiple processes (one process per core).
    Each process will process a range of slide numbers.
    """
    timer = Time()

    # how many processes to use
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    num_train_images = len(slides_images)
    if num_processes > num_train_images:
        num_processes = num_train_images
    images_per_process = num_train_images / num_processes

    print("Number of processes: " + str(num_processes))
    print("Number of images: " + str(num_train_images))

    # each task specifies a range of slides
    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        tasks.append((start_index, end_index, slides_images, dict_masked_pil_images, selected_tiles_dir, tile_size, desired_magnification, scale_factor))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(select_tiles_with_tissue_range, t))

    slides_tiles_coords = {}
    for result in results:
        (start_ind, end_ind, slides_tiles_coords_range) = result.get()
        slides_tiles_coords.update(slides_tiles_coords_range)
        if start_ind == end_ind:
            print("Done converting slide %d" % start_ind)
        else:
            print("Done converting slides %d through %d" % (start_ind, end_ind))

    print(">> Time to select tiles for all images (multiprocess): %s" % str(timer.elapsed()))

    return slides_tiles_coords


def select_tiles_with_tissue_range(start_index, end_index, slides_info, dict_masked_pil_images, selected_tiles_dir,
                                   tile_size, desired_magnification, scale_factor):
    slides_tiles_coords = {}
    for slide_num in range(start_index - 1, end_index):
        if os.path.isfile(os.path.join(selected_tiles_dir, slides_info[slide_num]['slide_name'] + '.npy')):
            print("Skipping slide " + slides_info[slide_num]['slide_name'])
        else:
            slides_tiles_coords[slides_info[slide_num]['slide_name']] = select_tiles_with_tissue_from_slide(slide_num, slides_info[slide_num], dict_masked_pil_images,
                                                                                                            selected_tiles_dir, tile_size, desired_magnification, scale_factor)
    return start_index, end_index, slides_tiles_coords


def select_tiles_with_tissue_from_slide(slide_num, slide_info, dict_masked_pil_images, selected_tiles_dir,
                                        tile_size, desired_magnification, scale_factor):
    # Initialize deep zoom generator for the slide
    image_dims = (slide_info['slide_width'], slide_info['slide_height'])
    image_name = slide_info['slide_name']
    slide = open_wsi(slide_info['slide_path'])
    dzg = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)

    # Find the deep zoom level corresponding to the requested magnification
    dzg_level_x = get_x_zoom_level(slide_info['highest_zoom_level'],
                                   slide_info['slide_magnification'], desired_magnification)
    # dzg_level_x = dzg.level_count - 1
    dzg_level_x_dims = dzg.level_dimensions[dzg_level_x]
    dzg_level_x_tile_coords = dzg.level_tiles[dzg_level_x]
    n_tiles = np.prod(dzg_level_x_tile_coords)

    # Calculate patch size in the mask
    dzg_downscaling = round(np.divide(image_dims, dzg_level_x_dims)[0])
    mask_patch_size = int(np.ceil(tile_size * (dzg_downscaling / scale_factor)))
    # Deep zoom generator for the mask
    dzg_mask = DeepZoomGenerator(openslide.ImageSlide(dict_masked_pil_images[image_name]), tile_size=mask_patch_size,
                                 overlap=0)
    dzg_mask_dims = dzg_mask.level_dimensions[dzg_mask.level_count - 1]
    dzg_mask_tile_coords = dzg_mask.level_tiles[dzg_mask.level_count - 1]
    dzg_mask_ntiles = np.prod(dzg_mask_tile_coords)

    if dzg_mask_tile_coords != dzg_level_x_tile_coords:
        print("Rounding error creates extra patches at the side(s) of the image " + slide_info['slide_name'])
        grid_coord = (min(dzg_mask_tile_coords[0], dzg_level_x_tile_coords[0]),
                      min(dzg_mask_tile_coords[1], dzg_level_x_tile_coords[1]))
        print("Ignoring the image border. Maximum tile coordinates: " + str(grid_coord))
        n_tiles = grid_coord[0] * grid_coord[1]
    else:
        grid_coord = dzg_level_x_tile_coords

    coords = []
    preds = [0] * n_tiles
    i = 0
    threshold = 0.90  # Threshold parameter indicating the proportion of the tile area that should be foreground (tissue content)
    # in order to be selected. It should range between 0 and 1.
    (cols, rows) = grid_coord

    for row in range(rows):
        for col in range(cols):
            mask_tile = dzg_mask.get_tile(dzg_mask.level_count - 1, (col, row))
            rgb_mask_tile = np.asarray(mask_tile)

            preds[i] = select_tile(rgb_mask_tile, threshold)

            tile = dzg.get_tile(dzg_level_x, (col, row))

            # we set the prediction to zero if the tile is not square -> we want only squared tiles
            if tile.size[0] != tile.size[1]:
                preds[i] = 0

            if preds[i] == 1:
                coord = (col, row)
                coords.append(coord)

            i += 1

    print(f"{slide_info['slide_name']}: slide num = {slide_num+1}, num tiles selected = {len(coords)}, zoom level = {dzg_level_x} (at %{desired_magnification}x), "
          f"num tiles = {n_tiles}, tile size = {tile_size}, mask tile size = {mask_patch_size},"
          f"slide dimensions = {image_dims}, slide dimensions (at %{desired_magnification}x) = {dzg_level_x_dims},"
          f"mask dimensions = {dzg_mask_dims}, mask num tiles = {dzg_mask_ntiles}")

    #np.save(os.path.join(selected_tiles_dir, slide_info['slide_name'] + '.npy'), coords)
    np.save(os.path.join(selected_tiles_dir, 'tmp_' + slide_info['slide_name'] + '.npy'), coords)

    os.rename(os.path.join(selected_tiles_dir, 'tmp_' + slide_info['slide_name'] + '.npy'),
              os.path.join(selected_tiles_dir, slide_info['slide_name'] + '.npy'))

    print(">> Tiles coords saved to \"%s\"" % selected_tiles_dir)

    return np.asarray(coords)


# Determine x Magnification Zoom Level
def get_x_zoom_level(highest_zoom_level, slide_magnification, desired_magnification):
    """
  Return the zoom level that corresponds to a x magnification.
  The generator can extract tiles from multiple zoom levels,
  downsampling by a factor of 2 per level from highest to lowest
  resolution.
  Args:

  Returns:
    Zoom level corresponding to a x magnification, or as close as
    possible.
  """
    # TODO: check if desired_magnification is proper value
    try:
        # `mag / desired_magnification` gives the downsampling factor between the slide's
        # magnification and the desired x magnification.
        # `(mag / desired_magnification) / 2` gives the zoom level offset from the highest
        # resolution level, based on a 2x downsampling factor in the
        # generator.
        if slide_magnification < 10:
            # slide magnification unknown
            return highest_zoom_level
        offset = math.floor((slide_magnification / desired_magnification) / 2)
        level = highest_zoom_level - offset
    except (ValueError, KeyError) as e:
        # In case the slide magnification level is unknown, just
        # use the highest resolution.
        level = highest_zoom_level
    return level


def select_tile(mask_patch, threshold):
    bg = np.all(mask_patch == np.array([0, 0, 0]), axis=2)
    bg_proportion = np.sum(bg) / bg.size

    if bg_proportion <= (1 - threshold):
        output = 1
    else:
        output = 0

    return output


# Normalize staining
def normalize_staining(sample, beta=0.15, alpha=1, light_intensity=255):
  """
  Normalize the staining of H&E histology slides.
  This function normalizes the staining of H&E histology slides.
  References:
    - Macenko, Marc, et al. "A method for normalizing histology slides
    for quantitative analysis." Biomedical Imaging: From Nano to Macro,
    2009.  ISBI'09. IEEE International Symposium on. IEEE, 2009.
      - http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    - https://github.com/mitkovetta/staining-normalization
  Args:
    sample_tuple: A (slide_num, sample) tuple, where slide_num is an
      integer, and sample is a 3D NumPy array of shape (H,W,C).
  Returns:
    A (slide_num, sample) tuple, where the sample is a 3D NumPy array
    of shape (H,W,C) that has been stain normalized.
  """
  # Setup.
  x = np.asarray(sample)
  h, w, c = x.shape
  x = x.reshape(-1, c).astype(np.float64)  # shape (H*W, C)

  # Reference stain vectors and stain saturations.  We will normalize all slides
  # to these references.  To create these, grab the stain vectors and stain
  # saturations from a desirable slide.

  # Values in reference implementation for use with eigendecomposition approach, natural log,
  # and `light_intensity=240`.
  #stain_ref = np.array([0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]).reshape(3,2)
  #max_sat_ref = np.array([1.9705, 1.0308]).reshape(2,1)

  # SVD w/ log10, and `light_intensity=255`.
  stain_ref = (np.array([0.54598845, 0.322116, 0.72385198, 0.76419107, 0.42182333, 0.55879629])
                 .reshape(3,2))
  max_sat_ref = np.array([0.82791151, 0.61137274]).reshape(2,1)

  # Convert RGB to OD.
  # Note: The original paper used log10, and the reference implementation used the natural log.
  #OD = -np.log((x+1)/light_intensity)  # shape (H*W, C)
  OD = -np.log10(x/light_intensity + 1e-8)

  # Remove data with OD intensity less than beta.
  # I.e. remove transparent pixels.
  # Note: This needs to be checked per channel, rather than
  # taking an average over all channels for a given pixel.
  OD_thresh = OD[np.all(OD >= beta, 1), :]  # shape (K, C)

  # Calculate eigenvectors.
  # Note: We can either use eigenvector decomposition, or SVD.
  #eigvals, eigvecs = np.linalg.eig(np.cov(OD_thresh.T))  # np.cov results in inf/nans
  U, s, V = np.linalg.svd(OD_thresh, full_matrices=False)

  # Extract two largest eigenvectors.
  # Note: We swap the sign of the eigvecs here to be consistent
  # with other implementations.  Both +/- eigvecs are valid, with
  # the same eigenvalue, so this is okay.
  #top_eigvecs = eigvecs[:, np.argsort(eigvals)[-2:]] * -1
  top_eigvecs = V[0:2, :].T * -1  # shape (C, 2)

  # Project thresholded optical density values onto plane spanned by
  # 2 largest eigenvectors.
  proj = np.dot(OD_thresh, top_eigvecs)  # shape (K, 2)

  # Calculate angle of each point wrt the first plane direction.
  # Note: the parameters are `np.arctan2(y, x)`
  angles = np.arctan2(proj[:, 1], proj[:, 0])  # shape (K,)

  # Find robust extremes (a and 100-a percentiles) of the angle.
  min_angle = np.percentile(angles, alpha)
  max_angle = np.percentile(angles, 100-alpha)

  # Convert min/max vectors (extremes) back to optimal stains in OD space.
  # This computes a set of axes for each angle onto which we can project
  # the top eigenvectors.  This assumes that the projected values have
  # been normalized to unit length.
  extreme_angles = np.array(
    [[np.cos(min_angle), np.cos(max_angle)],
     [np.sin(min_angle), np.sin(max_angle)]]
  )  # shape (2,2)
  stains = np.dot(top_eigvecs, extreme_angles)  # shape (C, 2)

  # Merge vectors with hematoxylin first, and eosin second, as a heuristic.
  if stains[0, 0] < stains[0, 1]:
    stains[:, [0, 1]] = stains[:, [1, 0]]  # swap columns

  # Calculate saturations of each stain.
  # Note: Here, we solve
  #    OD = VS
  #     S = V^{-1}OD
  # where `OD` is the matrix of optical density values of our image,
  # `V` is the matrix of stain vectors, and `S` is the matrix of stain
  # saturations.  Since this is an overdetermined system, we use the
  # least squares solver, rather than a direct solve.
  sats, _, _, _ = np.linalg.lstsq(stains, OD.T, rcond=-1)

  # Normalize stain saturations to have same pseudo-maximum based on
  # a reference max saturation.
  max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
  sats = sats / max_sat * max_sat_ref

  # Compute optimal OD values.
  OD_norm = np.dot(stain_ref, sats)

  # Recreate image.
  # Note: If the image is immediately converted to uint8 with `.astype(np.uint8)`, it will
  # not return the correct values due to the initital values being outside of [0,255].
  # To fix this, we round to the nearest integer, and then clip to [0,255], which is the
  # same behavior as Matlab.
  #x_norm = np.exp(OD_norm) * light_intensity  # natural log approach
  x_norm = 10**(-OD_norm) * light_intensity - 1e-8  # log10 approach
  x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
  x_norm = x_norm.astype(np.uint8)
  x_norm = x_norm.T.reshape(h,w,c)
  return x_norm


def preprocessing_images(slides_info, selected_tiles_dir, filter_info_path, scale_factor, tile_size, desired_magnification):
    slides_tiles_coords = {}
    if len(os.listdir(selected_tiles_dir)) == 0 or len(os.listdir(selected_tiles_dir)) < len(slides_info):
        # Apply filters to down scaled images
        print(">> Apply filters to down scaled images:")
        normal_segmented_images = multiprocess_apply_filters_to_wsi(slides_info, filter_info_path, scale_factor)

        print(">> Select from images the tiles with tissue:")
        slides_tiles_coords = multiprocess_select_tiles_with_tissue(slides_info, normal_segmented_images, selected_tiles_dir,
                                                                    tile_size, desired_magnification, scale_factor)
    else:
        print(">> Loading tiles coords from disk...")
        for slide_info in slides_info:
            slides_coords = np.load(os.path.join(selected_tiles_dir, slide_info['slide_name'] + '.npy'))
            slides_tiles_coords[slide_info['slide_name']] = slides_coords

    return slides_tiles_coords
