import base64
import os
import numpy as np
import multiprocessing
import openslide
from PIL import Image
from openslide.deepzoom import DeepZoomGenerator
from . import utils
from skimage import img_as_bool
from skimage.filters import median, gaussian
from skimage.morphology import disk
import config.images.config as cfg
import hashlib
import base64


def preprocessing_images(slides_info, selected_tiles_dir, filter_info_path, tiles_info_path, scale_factor, tile_size,
                         desired_magnification, images_dir, masked_images_dir):

    # Apply filters to down scaled images
    if len(os.listdir(masked_images_dir)) < len(slides_info):
        print(">> Apply filters to down scaled images:")
        multiprocess_apply_filters_to_wsi(slides_info, filter_info_path, scale_factor, images_dir, masked_images_dir)
    else:
        print(">> Masked images already available on disk")

    # Select tiles with tissue
    skipped = 0
    for slide in slides_info:
        if os.path.isfile(os.path.join(selected_tiles_dir, slide['slide_name'] + '.npy')):
            skipped +=1

    if skipped == len(slides_info):
        print(">> Selected tiles already available on disk")
    else:
        print(">> Select from images the tiles with tissue:")
        multiprocess_select_tiles_with_tissue(slides_info, masked_images_dir, selected_tiles_dir,
                                              tile_size, desired_magnification, scale_factor, tiles_info_path)


def multiprocess_apply_filters_to_wsi(slides_images, filter_info_path, scale_factor, images_dir, masked_images_dir):
    """
    Convert all WSI training slides to smaller images using multiple processes (one process per core).
    Each process will process a range of slide numbers.
    """
    timer = utils.Time()

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
        tasks.append((start_index, end_index, slides_images, scale_factor, images_dir,
                      masked_images_dir))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(apply_filters, t))

    filter_info = ""

    for result in results:
        (start_ind, end_ind, filter_info_range) = result.get()
        filter_info += filter_info_range
        if start_ind == end_ind:
            print("Done converting slide %d" % start_ind)
        else:
            print("Done converting slides %d through %d" % (start_ind, end_ind))

    print(">> Time to apply filters to all images (multiprocess): %s" % str(timer.elapsed()))

    images_info_file = open(filter_info_path, "w")
    images_info_file.write(filter_info)
    images_info_file.close()
    print(">> Filter info saved to \"%s\"\n" % filter_info_path)


def apply_filters(start_ind, end_ind, slide_images, scale_factor, images_dir, masked_images_dir):
    string = ""

    for slide_num in range(start_ind - 1, end_ind):
        scaled_image, scaled_w, scaled_h = utils.from_wsi_to_scaled_pillow_image(slide_images[slide_num]['slide_path'], scale_factor)
        scaled_image.save(os.path.join(images_dir, slide_images[slide_num]['slide_name'] + ".png"))
        info = apply_filters_to_image(slide_images[slide_num], scaled_image, masked_images_dir)
        string += info + '\n'

    return start_ind, end_ind, string


def apply_filters_to_image(slide_info, scaled_image, masked_images_dir, display=False):
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

    rgb = utils.pil_to_np_rgb(scaled_image)
    string += utils.np_info(rgb, "RGB", utils.Time().elapsed())
    string += '\n'
    if display:
        utils.display_img(rgb, "RGB")

    mask_no_red_pen = utils.filter_red_pen(rgb)
    mask_no_green_pen = utils.filter_green_pen(rgb)
    mask_no_blue_pen = utils.filter_blue_pen(rgb)
    mask_pens = mask_no_red_pen & mask_no_green_pen & mask_no_blue_pen
    rgb_pens = utils.mask_rgb(rgb, mask_pens)
    if display:
        utils.display_img(rgb_pens, "Pen filters")

    # from RGB to grayscale
    grayscale = utils.filter_rgb_to_grayscale(rgb)
    string += utils.np_info(grayscale, "Gray", utils.Time().elapsed())
    string += '\n'
    if display:
        utils.display_img(grayscale, "Grayscale")

    # otsu's adaptive thresholding
    # complement -> in order to have background values close to 0
    complement = utils.filter_complement(grayscale)
    string += utils.np_info(complement, "Complement", utils.Time().elapsed())
    string += '\n'
    if display:
        utils.display_img(complement, "Complement")

    otsu_mask = utils.filter_otsu_threshold(complement)
    string += utils.np_info(otsu_mask, "Otsu Threshold", utils.Time().elapsed())
    string += '\n'
    if display:
        utils.display_img(otsu_mask, "Compl. Otsu mask")

    median_filtering_otsu_mask = median(otsu_mask,
                                        disk(2))  # apply median filtering (radius = 2) on otsu mask (noise reduction)
    string += utils.np_info(median_filtering_otsu_mask, "MEDIAN FILTERING")
    string += '\n'
    if display:
        utils.display_img(median_filtering_otsu_mask, "Compl. Otsu mask(median)")

    blurring_otsu_mask = gaussian(median_filtering_otsu_mask, sigma=2)
    string += utils.np_info(blurring_otsu_mask, "BLURRING")
    string += '\n'
    if display:
        utils.display_img(blurring_otsu_mask, "Compl. Otsu mask(gaussian)")
    blurring_otsu_mask = img_as_bool(blurring_otsu_mask, force_copy=False)

    min_size_obj = 420  # smallest allowable object
    no_small_obj_otsu_mask = utils.filter_remove_small_objects(blurring_otsu_mask, min_size=min_size_obj)
    string += utils.np_info(no_small_obj_otsu_mask, "Remove Small Objs", utils.Time().elapsed())
    string += '\n'
    if display:
        utils.display_img(no_small_obj_otsu_mask, "Compl. Otsu mask(obj.)")
    no_small_obj_otsu_mask = img_as_bool(no_small_obj_otsu_mask, force_copy=False)

    no_small_holes_otsu_mask = utils.filter_remove_small_holes(no_small_obj_otsu_mask, max_size=100)
    string += utils.np_info(no_small_holes_otsu_mask, "Remove Small Holes", utils.Time().elapsed())
    string += '\n'
    if display:
        utils.display_img(no_small_holes_otsu_mask, "Compl. Otsu mask(holes)")

    no_small_holes_otsu_mask = img_as_bool(no_small_holes_otsu_mask, force_copy=False)
    segmented_image = utils.mask_rgb(rgb_pens, no_small_holes_otsu_mask)  # pixel wise and between the original image and the complementary of the otsu mask

    string += utils.np_info(segmented_image, "Mask RGB", utils.Time().elapsed())
    string += '\n'

    if display:
        utils.display_img(segmented_image, "Segmented image", bg=True)
    pil_segmented_image = utils.np_to_pil(segmented_image)

    # print("Image " + slide_info['slide_name'] + " masked")

    # np.save(os.path.join(path_to_save, slide_info['slide_name'] + ".npy"), segmented_image)
    pil_segmented_image.save(os.path.join(masked_images_dir, slide_info['slide_name'] + ".png"))
    print("Masked image saved to %s" % (os.path.join(masked_images_dir, slide_info['slide_name'] + ".png")))

    return string


def multiprocess_select_tiles_with_tissue(slides_images, masked_pil_images_dir, selected_tiles_dir,
                                          tile_size, desired_magnification, scale_factor, tiles_info_path):
    """
    Convert all WSI training slides to smaller images using multiple processes (one process per core).
    Each process will process a range of slide numbers.
    """
    timer = utils.Time()

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
        tasks.append((start_index, end_index, slides_images, masked_pil_images_dir, selected_tiles_dir, tile_size, desired_magnification, scale_factor))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(select_tiles_with_tissue_range, t))

    filter_info = ""
    for result in results:
        (start_ind, end_ind, filter_info_range) = result.get()
        filter_info += filter_info_range
        if start_ind == end_ind:
            print("Done converting slide %d" % start_ind)
        else:
            print("Done converting slides %d through %d" % (start_ind, end_ind))

    print(">> Time to select tiles for all images (multiprocess): %s" % str(timer.elapsed()))

    images_tiles_file = open(tiles_info_path, "w")
    images_tiles_file.write(filter_info)
    images_tiles_file.close()
    print(">> Tiles info saved to \"%s\"\n" % tiles_info_path)


def select_tiles_with_tissue_range(start_index, end_index, slides_info, masked_images_pil_dir, selected_tiles_dir,
                                   tile_size, desired_magnification, scale_factor):
    string = ""
    for slide_num in range(start_index - 1, end_index):
        info = select_tiles_with_tissue_from_slide(slides_info[slide_num], masked_images_pil_dir,
                                                   selected_tiles_dir, tile_size, desired_magnification, scale_factor)
        string += info + '\n'

    return start_index, end_index,string


def select_tiles_with_tissue_from_slide(slide_info, masked_images_pil_dir, selected_tiles_dir,
                                        tile_size, desired_magnification, scale_factor):
    # Initialize deep zoom generator for the slide
    image_dims = (slide_info['slide_width'], slide_info['slide_height'])
    slide = utils.open_wsi(slide_info['slide_path'])
    dzg = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)

    # Find the deep zoom level corresponding to the requested magnification
    dzg_level_x = utils.get_x_zoom_level(slide_info['highest_zoom_level'],
                                   slide_info['slide_magnification'], desired_magnification)
    # dzg_level_x = dzg.level_count - 1
    dzg_level_x_dims = dzg.level_dimensions[dzg_level_x]
    dzg_level_x_tile_coords = dzg.level_tiles[dzg_level_x]
    n_tiles = np.prod(dzg_level_x_tile_coords)

    # Calculate patch size in the mask
    dzg_downscaling = round(np.divide(image_dims, dzg_level_x_dims)[0])
    mask_patch_size = int(np.ceil(tile_size * (dzg_downscaling / scale_factor)))
    # Deep zoom generator for the mask
    pil_masked_image = Image.open(os.path.join(masked_images_pil_dir, slide_info['slide_name'] + ".png"))
    dzg_mask = DeepZoomGenerator(openslide.ImageSlide(pil_masked_image), tile_size=mask_patch_size,
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
    threshold = 0.90  # Threshold parameter indicating the proportion of the tile area that should be foreground (tissue content)
    # in order to be selected. It should range between 0 and 1.
    (cols, rows) = grid_coord

    for row in range(rows):
        for col in range(cols):
            mask_tile = dzg_mask.get_tile(dzg_mask.level_count - 1, (col, row))
            rgb_mask_tile = np.asarray(mask_tile)

            pred = utils.select_tile(rgb_mask_tile, threshold)

            tile = dzg.get_tile(dzg_level_x, (col, row))

            # we set the prediction to zero if the tile is not square -> we want only squared tiles
            if tile.size[0] != tile.size[1]:
                pred = 0

            if pred == 1:
                coord = (col, row)
                coords.append(coord)

    info = (f"{slide_info['slide_name']}: num tiles selected = {len(coords)}, slide magnification = {slide_info['slide_magnification']}, "
            f"highest zoom level = {slide_info['highest_zoom_level']}, zoom level = {dzg_level_x} (at %{desired_magnification}x), "
            f"num tiles = {n_tiles}, tile size = {tile_size}, mask tile size = {mask_patch_size},"
            f"slide dimensions = {image_dims}, slide dimensions (at %{desired_magnification}x) = {dzg_level_x_dims},"
            f"mask dimensions = {dzg_mask_dims}, mask num tiles = {dzg_mask_ntiles}")
    print(info)
    #np.save(os.path.join(selected_tiles_dir, slide_info['slide_name'] + '.npy'), coords)
    np.save(os.path.join(selected_tiles_dir, 'tmp_' + slide_info['slide_name'] + '.npy'), coords)

    os.rename(os.path.join(selected_tiles_dir, 'tmp_' + slide_info['slide_name'] + '.npy'),
              os.path.join(selected_tiles_dir, slide_info['slide_name'] + '.npy'))

    print(">> Tiles coords saved to \"%s\"" % os.path.join(selected_tiles_dir, slide_info['slide_name'] + '.npy'))
    return info


def hash_base64(_str):
    return base64.b64encode(hashlib.md5(_str.encode()).digest()).decode().rstrip("=")


def extract_tiles_on_disk(slides_info):
    for current_slide in slides_info:
        slide_name = current_slide['slide_name']
        print('Saving tiles of slide ', slide_name)
        slide = utils.open_wsi(current_slide['slide_path'])
        zoom = DeepZoomGenerator(slide, tile_size=224, overlap=0)

        dzg_level_x = utils.get_x_zoom_level(
            current_slide['highest_zoom_level'],
            current_slide['slide_magnification'],
            10)

        slide_tiles_coords = np.load(os.path.join(cfg.selected_coords_dir, slide_name + '.npy'))

        for index, coord in enumerate(slide_tiles_coords):
            tile = zoom.get_tile(dzg_level_x, (coord[0], coord[1]))
            np_tile = utils.normalize_staining(tile)
            save_path = cfg.selected_tiles_dir / (hash_base64(slide_name + str(index)).replace("/", "-") + '_' + slide_name + '.npy')
            np.save(save_path, np_tile)
