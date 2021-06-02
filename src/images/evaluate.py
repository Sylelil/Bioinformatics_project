import os
from pathlib import Path
from random import random

import tensorflow as tf

import config.images.config as cfg
import numpy as np
from tensorflow.keras.layers import Layer
from PIL import Image, ImageDraw, ImageFont
import openslide


class NoisyAnd(Layer):
    """Custom NoisyAND layer from the Deep MIL paper"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = 10  # fixed, controls the slope of the activation
        self.b = self.add_weight(name='b',
                                 shape=(1, input_shape[3]),
                                 initializer='uniform',
                                 trainable=True)
        super(NoisyAnd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        res = (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (
                tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
        return res

    def get_config(self):
        cfg = super(NoisyAnd, self).get_config()
        return cfg

    def from_config(self):
        return NoisyAnd(2)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]


def evaluate():
    path = Path("D:\\Bioinformatics_project\\results\\images\\selected_tiles\\tiles")
    files = list(os.listdir(path))
    np.random.shuffle(files)
    print("files len", len(files))
    # n_tile = -10
    already_seen = []
    for n_tile in range(len(files)):
        slide_name = files[n_tile].split("_")[1]
        if slide_name in already_seen:
            continue
        already_seen.append(slide_name)

        selected_file = files[n_tile]

        truth = 1 if selected_file.endswith("1.npy") else 0
        truth_label = "tumor" if truth == 1 else "normal"

        # print("Selected file = ", selected_file)

        filtered_tiles = list(filter(lambda x: x.split("_")[1] == slide_name, files))

        model = tf.keras.models.load_model(Path("cp2.h5"), custom_objects={"NoisyAnd": NoisyAnd})
        #model = tf.keras.models.Model

        total_tumor = 0
        images = []
        predictions = []
        all_imgs = []
        for tile in filtered_tiles:
            all_imgs.append(np.load(path / tile))
        np_tile = np.concatenate(np.expand_dims(all_imgs, axis=0), axis=0)
        norm_np_tile = np_tile / 255.0
        print(np_tile.shape)
        el = model.predict(norm_np_tile)
        print(selected_file, '\t', truth, '\t', np.average(el))
        # prediction = 0 if el < 0.5 else 1
        # total_tumor += prediction
        # predictions.append(el)
        # img = Image.fromarray(np_tile)
        # draw = ImageDraw.Draw(img)
        # label = "tumor" if prediction == 1 else "normal"
        # draw.text((80, 0), "Label = " + label, (20, 40, 210))
        # images.append(img)

        # for tile in filtered_tiles:
        #     np_tile = np.load(path / tile)
        #     norm_np_tile = np_tile / 255.0
        #     el = model.predict(np.expand_dims(norm_np_tile, axis=0))
        #     prediction = 0 if el < 0.5 else 1
        #     total_tumor += prediction
        #     predictions.append(el)
        #     img = Image.fromarray(np_tile)
        #     draw = ImageDraw.Draw(img)
        #     label = "tumor" if prediction == 1 else "normal"
        #     draw.text((80,0), "Label = " + label, (20,40,210))
        #     images.append(img)

        # grid = pil_grid(images, 10)
        # grid.save("D:\\Bioinformatics_project\\results\\grids\\" + selected_file.replace(".npy", ".png"))
        # percentage_tumor = 1.0 * total_tumor / len(filtered_tiles)
        # print("file ", selected_file, " was ", truth_label, " and it got ", percentage_tumor)
        # print("or ", np.average(np.array(predictions)))


def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * (1+n_horiz), [0] * (1+n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid


if __name__ == '__main__':

    # p = Path("D:\\Bioinformatics_project\\datasets\\images\\ec45ae69-abe2-448d-892e-e28083b3da61\\8e87ddca-f43e-4c6f-9d5b-430b666a7f6c_1.svs")
    # o = openslide.OpenSlide(str(p))
    # o.get_thumbnail((1024, 1024)).save("C:\\users\\crist\\file.png")
    evaluate()
