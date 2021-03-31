import os
import openslide
from openslide import OpenSlideError
import datetime
import math
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import skimage.morphology as sk_morphology
import skimage.filters as sk_filters


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

    if output_type == "bool":
        pass
    elif output_type == "float":
        otsu = otsu.astype(float)
    else:
        otsu = otsu.astype("uint8") * 255

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


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=False, overmask_thresh=95, output_type="uint8"):
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


def display_img(np_img, text=None, size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.
    Args:
      np_img: Image as a NumPy array.
      text: The text to add to the image.
      font_path: The path to the font to use.
      size: The font size
      color: The font color
      background: The background color
      border: The border color
      bg: If True, add rectangle background behind text
    """
    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == 'L':
        result = result.convert('RGB')
    draw = ImageDraw.Draw(result)
    if text is not None:
        # arial si spacca male al di fuori di Windows
        # font = ImageFont.truetype("arial.ttf", size)

        if bg:
            (x, y) = draw.textsize(text) #, font)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color) #, font=font)
    result.show()


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


def create_summary_pil_img(np_img, title_area_height, row_tile_size, col_tile_size, num_row_tiles, num_col_tiles):
    """
    reate a PIL summary image including top title area and right side and bottom padding.
    Args:
      np_img: Image as a NumPy array.
      title_area_height: Height of the title area at the top of the summary image.
      row_tile_size: The tile size in rows.
      col_tile_size: The tile size in columns.
      num_row_tiles: The number of row tiles.
      num_col_tiles: The number of column tiles.
    Returns:
      Summary image as a PIL image. This image contains the image data specified by the np_img input and also has
      potentially a top title area and right side and bottom padding.
    """
    r = row_tile_size * num_row_tiles + title_area_height
    c = col_tile_size * num_col_tiles
    summary_img = np.zeros([r, c, np_img.shape[2]], dtype=np.uint8)
    # add gray edges so that tile text does not get cut off
    summary_img.fill(120)
    # color title area white
    summary_img[0:title_area_height, 0:summary_img.shape[1]].fill(255)
    summary_img[title_area_height:np_img.shape[0] + title_area_height, 0:np_img.shape[1]] = np_img
    summary = np_to_pil(summary_img)
    return summary

