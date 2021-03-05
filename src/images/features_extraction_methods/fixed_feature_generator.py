import sys

import numpy
import openslide
import openslide.deepzoom
from pathlib import Path
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image
import colorcorrect
from colorcorrect.util import from_pil, to_pil
from colorcorrect.algorithm import stretch
import os
import itertools
import multiprocessing
from openslide.deepzoom import DeepZoomGenerator
from src.images import utils
from src.images.utils import Time


def save_numpy_features(slide_info, slides_tiles_coords, tile_size, desired_magnification, path_to_save):
    model = ResNet50(weights='imagenet', include_top=True)
    model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)

    slide = utils.open_wsi(slide_info['slide_path'])
    zoom = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0)

    # Find the deep zoom level corresponding to the requested magnification
    dzg_level_x = utils.get_x_zoom_level(slide_info['highest_zoom_level'], slide_info['slide_magnification'],
                                         desired_magnification)

    tiles = []
    for coord in slides_tiles_coords:
        tile = zoom.get_tile(dzg_level_x, (coord[0], coord[1]))
        np_tile = utils.normalize_staining(tile)
        tiles.append(np_tile)

    tiles = np.asarray(tiles)
    #print(f'tiles: {tiles.shape}')
    tiles = preprocess_input(tiles)  # Preprocesses a tensor or Numpy array encoding a batch of images

    X = model.predict(tiles, batch_size=32, verbose=1)
    X = np.concatenate([slides_tiles_coords, X], axis=1)
    np.save(os.path.join(path_to_save, slide_info['slide_name'] + '_' + str(dzg_level_x) + '.npy'), X)


def extract_tile_features(level, coord, zoom):
    print(f"Extracting coords {coord} of {zoom.tile_count}...")

    tile = zoom.get_tile(level, coord)
    #tile = Image.fromarray(tile)
    #if numpy.average(numpy.array(tile)) != 0:
    try:
        tile = to_pil(stretch(from_pil(tile)))
    except:
        print("error in stretching")
    tile = np.array(tile)
    return tile


def get_all_slides(lookup_dir):
    images_list = []  # formato: {"dir": abc/def/sani/ecc, "file": file.svs

    for _dir in os.listdir(lookup_dir):
        current_dir = os.path.join(lookup_dir, _dir)
        if os.path.isdir(current_dir):
            for file in os.listdir(current_dir):
                if file.endswith('.svs'):
                    svs_dict = {
                        'dir': current_dir,
                        'file': file
                    }
                    images_list.append(svs_dict)
    return images_list


def save_numpy_features_range(start_ind, end_ind, slides_info, tile_size, images_save_dir,
                              slides_tiles_coords, desired_magnification):
    for slide_num in range(start_ind - 1, end_ind):
        save_numpy_features(slides_info[slide_num], slides_tiles_coords[slides_info[slide_num]['slide_name']],
                            tile_size, desired_magnification, images_save_dir)
    return start_ind, end_ind


def multiprocess_save_numpy_features(images_info, slides_tiles_coords, numpy_features_dir,
                                     tile_size, desired_magnification):
    timer = Time()

    # how many processes to use
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)

    num_train_images = len(images_info)
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
        tasks.append((start_index, end_index, images_info, tile_size, numpy_features_dir,
                      slides_tiles_coords, desired_magnification))
        if start_index == end_index:
            print("Task #" + str(num_process) + ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " + str(start_index) + " to " + str(end_index))

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(save_numpy_features_range, t))

    for result in results:
        (start_ind, end_ind) = result.get()
        if start_ind == end_ind:
            print("Done extracting features from slide %d" % start_ind)
        else:
            print("Done extracting features from slide %d through %d" % (start_ind, end_ind))

    print(">> Time to extract features from all images (multiprocess): %s" % str(timer.elapsed()))


def fixed_feature_generator(slides_tiles_coords, images_info, numpy_features_dir, tile_size, desired_magnification):

    multiprocess_save_numpy_features(images_info, slides_tiles_coords, numpy_features_dir,
                                     tile_size, desired_magnification)
