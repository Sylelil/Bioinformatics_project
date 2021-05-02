import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
import multiprocessing
from openslide.deepzoom import DeepZoomGenerator
import config.images.config as cfg
from .. import utils


def save_numpy_features(slide_info, path_to_save, selected_tiles_dir):
    print(">> Image %s:" % (slide_info['slide_name']))
    print(">> Loading pretrained model...")
    model = ResNet50(weights='imagenet', include_top=True)
    model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)

    print(">> Pretrained model loaded")

    slide = utils.open_wsi(slide_info['slide_path'])
    zoom = DeepZoomGenerator(slide, tile_size=cfg.TILE_SIZE, overlap=cfg.OVERLAP)

    # Find the deep zoom level corresponding to the requested magnification
    dzg_level_x = utils.get_x_zoom_level(slide_info['highest_zoom_level'], slide_info['slide_magnification'],
                                         cfg.DESIRED_MAGNIFICATION)

    tiles = []
    print(">> Getting tiles..")
    slide_tiles_coords = np.load(os.path.join(selected_tiles_dir, slide_info['slide_name'] + '.npy'))
    for coord in slide_tiles_coords:
        tile = zoom.get_tile(dzg_level_x, (coord[0], coord[1]))
        np_tile = utils.normalize_staining(tile)
        tiles.append(np_tile)

    print(">> tiles loaded")
    print("Num tiles = %d" % len(tiles))

    tiles = np.asarray(tiles)
    print(tiles.shape)

    # print(f'tiles: {tiles.shape}')
    print(">> Preprocessing numpy array of tiles...")
    tiles = preprocess_input(tiles)  # Preprocesses a tensor or Numpy array encoding a batch of images
    print(">> Numpy array of tiles preprocessed")

    print(">> Prediction...")
    X = model.predict(tiles, batch_size=32, verbose=1)
    print(X.shape) # (num tiles, 2048)
    print(">> Done")
    X = np.concatenate([slide_tiles_coords, X], axis=1)
    np.save(os.path.join(path_to_save, slide_info['slide_name'] + '.npy'), X)


def save_numpy_features_range(start_ind, end_ind, slides_info, images_save_dir, selected_tiles_dir):
    for slide_num in range(start_ind - 1, end_ind):
        if os.path.isfile(os.path.join(images_save_dir, slides_info[slide_num]['slide_name'] + '.npy')):
            print("Skipping slide " + slides_info[slide_num]['slide_name'])
        else:
            save_numpy_features(slides_info[slide_num], images_save_dir, selected_tiles_dir)
    return start_ind, end_ind


def multiprocess_save_numpy_features(images_info, numpy_features_dir, selected_tiles_dir):
    timer = utils.Time()

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
        tasks.append((start_index, end_index, images_info, numpy_features_dir, selected_tiles_dir))
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


def fixed_feature_generator(images_info, numpy_features_dir, selected_tiles_dir):

    if len(os.listdir(numpy_features_dir)) == 0 or len(os.listdir(numpy_features_dir)) < len(images_info):
        if cfg.USE_GPU:
            gpus = tf.config.experimental.list_physical_devices('GPU')

            if gpus:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            for slide_info in images_info:
                if os.path.isfile(os.path.join(numpy_features_dir, slide_info['slide_name'] + '.npy')):
                    print("Skipping slide " + slide_info['slide_name'])
                else:
                    save_numpy_features(slide_info, numpy_features_dir, selected_tiles_dir)
        else:
            with tf.device('/cpu:0'):
                multiprocess_save_numpy_features(images_info, numpy_features_dir, selected_tiles_dir)
