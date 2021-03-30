import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from openslide.deepzoom import DeepZoomGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from src.images import utils
from sklearn.metrics import classification_report
import random

def fine_tuning(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir):

    train_gen = feed_slides_generator(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir, 3, mode="train")
    eval_gen = feed_slides_generator(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir, 3, mode="eval")
    test_gen = feed_slides_generator(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir, 3, mode="eval")

    train_len = len(normal_slides_info) + len(tumor_slides_info) - 5
    val_len = 5

    model = ResNet50(weights='imagenet', include_top=True)
    model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)

    # print(model.summary())

    baseModel = ResNet50(weights='imagenet', include_top=True)

    x = baseModel.output
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    model = Model(inputs=baseModel.inputs, outputs=x)

    #print(model.summary())

    for layer in baseModel.layers:
        layer.trainable = False

    print("[INFO] compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    print("[INFO] training head...")
    H = model.fit(
        x=train_gen,
        steps_per_epoch= train_len // 3,
        validation_data=eval_gen,
        validation_steps= val_len // 3,
        epochs=1)

    predIdxs = model.predict(x=test_gen, steps=(val_len // 3) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)

    # print(classification_report({}, predIdxs))
    plot_training(H, 50)


def feed_slides_generator(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir, batch_size, mode='train'):
    slides_info = [*normal_slides_info, *tumor_slides_info]
    random.shuffle(slides_info)

    train_slides = filter(lambda s: s['slide_name'].startswith('b') or s['slide_name'].startswith('f'), slides_info)
    eval_slides = filter(lambda s: not s['slide_name'].startswith('b') and not s['slide_name'].startswith('f'), slides_info)

    slides_info = list(train_slides) if mode == 'train' else list(eval_slides)

    slide_num = 0

    while True:
        data = np.array([]).reshape((0, 224, 224, 3))
        labels = []
        index = 0
        while index < 3:
            current_slide = slides_info[slide_num]
            slide = utils.open_wsi(current_slide['slide_path'])
            zoom = DeepZoomGenerator(slide, tile_size=224, overlap=0)

            # Find the deep zoom level corresponding to the requested magnification
            dzg_level_x = utils.get_x_zoom_level(current_slide['highest_zoom_level'], current_slide['slide_magnification'],
                                                 10)

            selected_tiles_dir = normal_selected_tiles_dir if current_slide['label'] == 0 else tumor_selected_tiles_dir
            tiles = []
            print(">> Getting tiles..")
            slide_tiles_coords = np.load(os.path.join(selected_tiles_dir, current_slide['slide_name'] + '.npy'))
            for coord in slide_tiles_coords:
                # print(coord)
                tile = zoom.get_tile(dzg_level_x, (coord[0], coord[1]))
                np_tile = utils.normalize_staining(tile)
                # print(np_tile.shape)
                tiles.append(np_tile)

            print(">> tiles loaded")
            print("Num tiles = %d" % len(tiles))

            data = np.concatenate((data, tiles))
            #label = "normal" if current_slide["label"] == 0 else "tumor"
            labels = [*labels, *(current_slide["label"] for s in range(len(tiles)))]
            slide_num += 1
            index += 1
            if slide_num == len(slides_info):
                slide_num = 0

        print("Shape: ", data.shape)
        yield np.array(data), np.array(labels).reshape(-1, 1)


def plot_training(H, N):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


if __name__ == '__main__':
    fine_tuning()

# 1: separare test, val e train
# 2: capire come estrarre le tile, feedarle
# 3: froze / unfroze
# 4: rileggere il capitolo prediction, e se facessimo il feed delle tile, poi facciamo la prediction di un'immagine dividendola in tile e si inferisce la soglia
# poi possiamo utilizzare lo stesso procedimento per etichettare le singole tile
