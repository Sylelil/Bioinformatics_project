import os
import openslide
from openslide import OpenSlideError
import datetime
import math
import numpy as np
import PIL
from PIL import Image, ImageDraw
import skimage.morphology as sk_morphology
import skimage.filters as sk_filters


class Time:
    """
        Description: Class for displaying elapsed time
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
       :param file_path: Path
            path to wsi
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
        Description: retrieve WSI magnification
        :param slide: OpenSlide object
        :return: wsi magnification
    """
    return int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])


def get_wsi_highest_zoom_level(generator):
    """
        Description: retrieve highest zoom level for slide, given DeepZoomGenerator wrapper
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
        Description: Convert a PIL Image to a NumPy array.
                     Note that RGB PIL (w, h) -> NumPy (h, w, 3).
        :param pil_img: The PIL Image.
        :return: The PIL image converted to a NumPy array.
    """
    t = Time()
    rgb = np.asarray(pil_img)
    return rgb


def from_wsi_to_scaled_pillow_image(file_path, scale_factor):
    """
       Description : Convert a WSI training slide to a scaled-down PIL image.
       :param scale_factor: scale factor
       :param file_path: path to WSI
       :return: Tuple consisting of scaled-down PIL image, new width, and new height
    """
    # print("Opening Slide: %s" % file_path)
    slide = open_wsi(file_path)
    large_w, large_h = slide.dimensions

    new_w = math.floor(large_w / scale_factor)
    new_h = math.floor(large_h / scale_factor)

    # Return the best level for displaying the given downsample
    level = slide.get_best_level_for_downsample(scale_factor)

    # Return an RGBA Image containing the contents of the specified region
    # location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")

    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, new_w, new_h


def np_info(np_arr, name=None, elapsed=None):
    """
        Description: Display information (shape, type, max, min, etc) about a NumPy array.
        :param np_arr: The NumPy array.
        :param name: The (optional) name of the array.
        :param elapsed: The (optional) time elapsed to perform a filtering operation.
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
      Description: Convert an RGB NumPy array to a grayscale NumPy array.
                   Shape (h, w, c) to (h, w).
      :param np_img: RGB Image as a NumPy array.
      :param output_type: Type of array to return (float or uint8)
      :return: Grayscale image as NumPy array with shape (h, w).
    """

    # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    if output_type != "float":
        grayscale = grayscale.astype("uint8")
    return grayscale


def filter_otsu_threshold(np_img, output_type="uint8"):
    """
      Description: Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
      :param np_img: Image as a NumPy array.
      :param output_type: Type of array to return (bool, float, or uint8).
      :return: NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
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
       Description: Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
       :param rgb: RGB image as a NumPy array.
       :param mask: An image mask to determine which pixels in the original image should be displayed.
       :return: NumPy array representing an RGB image with mask applied.
    """
    result = rgb * np.dstack([mask, mask, mask])
    return result


def filter_complement(np_img, output_type="uint8"):
    """
      Description: Obtain the complement of an image as a NumPy array.
      :param np_img: Image as a NumPy array.
      :param output_type: Type of array to return (float or uint8).
      :return: Complement image as Numpy array.
    """

    if output_type == "float":
        complement = 1.0 - np_img
    else:
        complement = 255 - np_img
    return complement


def filter_red_pen(rgb, output_type="bool"):
    """
        Description: Create a mask to filter out red pen marks from a slide.
        :param rgb: RGB image as a NumPy array.
        :param output_type: Type of array to return (bool, float, or uint8).
        :return: NumPy array representing the mask.
    """

    result = filter_red(rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90) & \
             filter_red(rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30) & \
             filter_red(rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105) & \
             filter_red(rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125) & \
             filter_red(rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145) & \
             filter_red(rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70) & \
             filter_red(rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150) & \
             filter_red(rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65) & \
             filter_red(rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45) & \
             filter_red(rgb, red_lower_thresh=232, green_upper_thresh=186, blue_upper_thresh=203) & \
             filter_red(rgb, red_lower_thresh=224, green_upper_thresh=149, blue_upper_thresh=180) & \
             filter_red(rgb, red_lower_thresh=239, green_upper_thresh=229, blue_upper_thresh=232) & \
             filter_red(rgb, red_lower_thresh=226, green_upper_thresh=151, blue_upper_thresh=185)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255

    return result


def filter_red(rgb, red_lower_thresh, green_upper_thresh, blue_upper_thresh, output_type="bool"):
    """
        Description: Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
        red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

        :param rgb: RGB image as a NumPy array.
        :param red_lower_thresh: Red channel lower threshold value.
        :param green_upper_thresh: Green channel upper threshold value.
        :param blue_upper_thresh: Blue channel upper threshold value.
        :param output_type: Type of array to return (bool, float, or uint8).
        :return: NumPy array representing the mask.
    """

    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255

    return result


def filter_green(rgb, red_upper_thresh, green_lower_thresh, blue_lower_thresh, output_type="bool"):
    """
    Description: Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.

    :param rgb: RGB image as a NumPy array.
    :param red_upper_thresh: Red channel upper threshold value.
    :param green_lower_thresh: Green channel lower threshold value.
    :param blue_lower_thresh: Blue channel lower threshold value.
    :param output_type: Type of array to return (bool, float, or uint8).
    :return: NumPy array representing the mask.
    """
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh

    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255

    return result


def filter_green_pen(rgb, output_type="bool"):
    """
    Description: Create a mask to filter out green pen marks from a slide.
    :param rgb: RGB image as a NumPy array.
    :param output_type: Type of array to return (bool, float, or uint8).
    :return: NumPy array representing the mask.
    """

    result = filter_green(rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140) & \
             filter_green(rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110) & \
             filter_green(rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100) & \
             filter_green(rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60) & \
             filter_green(rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210) & \
             filter_green(rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225) & \
             filter_green(rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200) & \
             filter_green(rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20) & \
             filter_green(rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40) & \
             filter_green(rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35) & \
             filter_green(rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60) & \
             filter_green(rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105) & \
             filter_green(rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180) & \
             filter_green(rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150) & \
             filter_green(rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195) & \
             filter_green(rgb, red_upper_thresh=84, green_lower_thresh=69, blue_lower_thresh=88) & \
             filter_green(rgb, red_upper_thresh=110, green_lower_thresh=137, blue_lower_thresh=145) & \
             filter_green(rgb, red_upper_thresh=130, green_lower_thresh=166, blue_lower_thresh=165) & \
             filter_green(rgb, red_upper_thresh=105, green_lower_thresh=149, blue_lower_thresh=150) & \
             filter_green(rgb, red_upper_thresh=123, green_lower_thresh=122, blue_lower_thresh=136)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    return result


def filter_blue(rgb, red_upper_thresh, green_upper_thresh, blue_lower_thresh, output_type="bool",
                display_np_info=False):
    """
    Description: Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

    :param rgb: RGB image as a NumPy array.
    :param red_upper_thresh: Red channel upper threshold value.
    :param green_upper_thresh: Green channel upper threshold value.
    :param blue_lower_thresh: Blue channel lower threshold value.
    :param output_type: Type of array to return (bool, float, or uint8).
    :param display_np_info: If True, display NumPy array info and filter time.
    :return: NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh

    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255

    return result


def filter_blue_pen(rgb, output_type="bool"):
    """
    Description: Create a mask to filter out blue pen marks from a slide.

    :param rgb: RGB image as a NumPy array.
    :param output_type: Type of array to return (bool, float, or uint8).
    :return: NumPy array representing the mask.
    """
    t = Time()
    result = filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190) & \
             filter_blue(rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200) & \
             filter_blue(rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230) & \
             filter_blue(rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210) & \
             filter_blue(rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160) & \
             filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130) & \
             filter_blue(rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180) & \
             filter_blue(rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85) & \
             filter_blue(rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65) & \
             filter_blue(rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140) & \
             filter_blue(rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120) & \
             filter_blue(rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255

    return result


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=False, overmask_thresh=95, output_type="uint8"):
    """
    Description: Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
    is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
    reduce the amount of masking that this filter performs.

    :param np_img: Image as a NumPy array of type bool.
    :param min_size: Minimum size of small object to remove.
    :param avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    :param overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    :param output_type: Type of array to return (bool, float, or uint8).
    :return: NumPy array (bool, float, or uint8).
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
    Description: Filter image to remove small holes less than a particular size.

    :param np_img: Image as a NumPy array of type bool.
    :param max_size: Remove small holes below this size.
    :param output_type: Type of array to return (bool, float, or uint8).
    :return: NumPy array (bool, float, or uint8).
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
    Description: Convert a NumPy array to a PIL Image.

    :param np_img: The image represented as a NumPy array.
    :return: the NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def mask_percent(np_img):
    """
      Description: Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

      :param np_img: Image as a NumPy array.
      :return: The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage


def display_img(np_img, text=None, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
    """
        Description: Convert a NumPy array to a PIL image, add text to the image, and display the image.

        :param np_img: Image as a NumPy array.
        :param text: The text to add to the image.
        :param color: The font color
        :param background: The background color
        :param border: The border color
        :param bg: If True, add rectangle background behind text
    """
    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == 'L':
        result = result.convert('RGB')
    draw = ImageDraw.Draw(result)
    if text is not None:

        if bg:
            (x, y) = draw.textsize(text)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color)
    result.show()


# Determine x Magnification Zoom Level
def get_x_zoom_level(highest_zoom_level, slide_magnification, desired_magnification):
    """
      Description: Return the zoom level that corresponds to a x magnification.
                   The generator can extract tiles from multiple zoom levels,
                   downsampling by a factor of 2 per level from highest to lowest resolution.
      :param highest_zoom_level: int
            highest Deep zoom slide level
      :param slide_magnification: int
            slide magnification
      :param desired_magnification: int
            x magnification
      :return: int
            Zoom level corresponding to a x magnification, or as close as possible.
    """
    try:
        # `slide_magnification / desired_magnification` gives the downsampling factor between the slide's
        # magnification and the desired x magnification.
        # `(slide_magnification / desired_magnification) / 2` gives the zoom level offset from the highest
        # resolution level, based on a 2x downsampling factor in the generator.
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
    """
        Description: Determines if a mask tile contains a percentage of foreground above a threshold.

        :param mask_patch: Numpy array for the current mask tile.
        :param threshold: Float indicating the minimum foreground content [0, 1] in the patch to select the tile.
        :return: Integer [0/1] indicating if the tile has been selected or not.
        """
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
    Description: Normalize the staining of H&E histology slides.
                 This function normalizes the staining of H&E histology slides.
    References:
      - Macenko, Marc, et al. "A method for normalizing histology slides
      for quantitative analysis." Biomedical Imaging: From Nano to Macro,
      2009.  ISBI'09. IEEE International Symposium on. IEEE, 2009.
        - http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
      - https://github.com/mitkovetta/staining-normalization
    :param sample: is a 3D NumPy array of shape (H,W,C).
    :return: a 3D NumPy array of shape (H,W,C) that has been stain normalized.
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