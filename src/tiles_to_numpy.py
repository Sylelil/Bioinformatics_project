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


def save_numpy_features(dir, slidename, path_to_save):
    model = ResNet50(weights='imagenet', include_top=True)
    model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)

    slide = openslide.OpenSlide(os.path.join(dir, slidename))
    zoom = openslide.deepzoom.DeepZoomGenerator(slide, tile_size=224, overlap=0)
    levels = (zoom.level_count - 1, zoom.level_count - 2)
    level = levels[1]
    save_path = os.path.join(path_to_save, slidename.split('.')[0] + str(level) + '.npy')
    if os.path.exists(save_path):
        print(f'File {save_path} exist. Skipping')
        return
    
    print(f'size is {zoom.level_dimensions[level]}')
    cols, rows = zoom.level_tiles[level]
    print(f'cols is {cols}, rows is {rows}')
    print(slide.properties)
    print(f'region 1 is {zoom.get_tile_coordinates(level, (0, 0))}')
    print(f'region 2 is {zoom.get_tile_coordinates(level, (cols-1, rows-1))}')
    coords = []
    for col in range(cols-1):
        for row in range(rows-1):
            coord = (col, row)
            coords.append(coord)

    #coords = coords[:-100]
    print(f'number of coords: {len(coords)}')

    tiles = np.array([extract_tile_features(level, coord, zoom) for coord in tqdm(coords)])
    tiles = preprocess_input(tiles)

    X = model.predict(tiles, batch_size=32, verbose=1)
    X = np.concatenate([coords, X], axis=1)
    np.save(os.path.join(path_to_save, slidename.split('.')[0] + str(level) + '.npy'), X)


def extract_tile_features(level, coord, zoom):
    print(f"Extracting coords {coord} of {zoom.tile_count}...")

    tile = zoom.get_tile(level, coord)
    tile.save(os.path.join(Path("tiles"), "%d_%d.png" % (coord[0], coord[1])))
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


def main():
    '''
    images_folder = Path('datasets') / 'images'
    classes = ('normal', 'tumor')

    normal_images_dir = os.path.join(images_folder, classes[0])
    tumor_images_dir = os.path.join(images_folder, classes[1])
    '''

    normal_images_list = get_all_slides(sys.argv[1])
    tumor_images_list = get_all_slides(sys.argv[2])

    normal_images_save_dir = Path('generated') / 'numpy_normal'
    tumor_images_save_dir = Path('generated') / 'numpy_tumor'

    if not os.path.exists(Path('generated')):
        os.mkdir('generated')
    if not os.path.exists(normal_images_save_dir):
        os.mkdir(normal_images_save_dir)
    if not os.path.exists(tumor_images_save_dir):
        os.mkdir(tumor_images_save_dir)

    for item in normal_images_list:
        save_numpy_features(item['dir'], item['file'], normal_images_save_dir)
        break
    '''
    for item in tumor_images_list:
        save_numpy_features(item['dir'], item['file'], tumor_images_save_dir)
    '''

if __name__ == '__main__':
    main()
