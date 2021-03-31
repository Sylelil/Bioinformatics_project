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


    #TODO: move to utils function
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 500
    train_gen = feed_slides_generator(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir,
                                      tumor_selected_tiles_dir, batch_size, mode="train")
    eval_gen = feed_slides_generator(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir,
                                     tumor_selected_tiles_dir, batch_size, mode="eval")
    test_gen = feed_slides_generator(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir,
                                     tumor_selected_tiles_dir, batch_size, mode="eval")

    train_len = get_train_ds_len(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir)
    val_len = get_eval_ds_len(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir)

    # model = ResNet50(weights='imagenet', include_top=True)
    # model = Model(inputs=model.inputs, outputs=model.get_layer('avg_pool').output)

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
        steps_per_epoch=train_len // batch_size,
        validation_data=eval_gen,
        validation_steps=val_len // batch_size,
        epochs=10)

    model.save("fine_tuned")

    predIdxs = model.predict(x=test_gen, steps=(val_len // batch_size) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)

    # print(classification_report({}, predIdxs))
    plot_training(H, 10)


def feed_slides_generator(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir,
                          tumor_selected_tiles_dir, batch_size, mode='train'):
    slides_info = [*normal_slides_info, *tumor_slides_info]
    random.shuffle(slides_info)

    train_slides = filter(lambda s: s['slide_name'].startswith('b') or s['slide_name'].startswith('f'), slides_info)
    eval_slides = filter(lambda s: not s['slide_name'].startswith('b') and not s['slide_name'].startswith('f'), slides_info)
    slides_info = list(train_slides) if mode == 'train' else list(eval_slides)
    slide_num = 0
    data = np.array([]).reshape((0, 224, 224, 3))

    labels = []

    num_tiles_prefetch = 3

    while True:
        tiles = []
        for _ in range(num_tiles_prefetch):
            current_slide = slides_info[slide_num]
            slide = utils.open_wsi(current_slide['slide_path'])
            zoom = DeepZoomGenerator(slide, tile_size=224, overlap=0)

            # Find the deep zoom level corresponding to the requested magnification
            dzg_level_x = utils.get_x_zoom_level(current_slide['highest_zoom_level'], current_slide['slide_magnification'],
                                                 10)
            selected_tiles_dir = normal_selected_tiles_dir if current_slide['label'] == 0 else tumor_selected_tiles_dir
            print(">> Getting tiles..")
            slide_tiles_coords = np.load(os.path.join(selected_tiles_dir, current_slide['slide_name'] + '.npy'))
            labels = [*labels, *(current_slide["label"] for s in range(len(tiles)))]

            for coord in slide_tiles_coords:
                tile = zoom.get_tile(dzg_level_x, (coord[0], coord[1]))
                np_tile = utils.normalize_staining(tile)
                tiles.append(np_tile)

        while len(data) > batch_size:
            yield np.array(data[batch_size:]), np.array(labels[batch_size:]).reshape(-1, 1)
            del data[batch_size:]
            del labels[batch_size:]

        slide_num += num_tiles_prefetch
        if slide_num > len(slides_info):
            slide_num = 0


def get_train_ds_len(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir):
    slides_info = [*normal_slides_info, *tumor_slides_info]
    train_slides = list(filter(lambda s: s['slide_name'].startswith('b') or s['slide_name'].startswith('f'), slides_info))

    acc = 0
    for slide in train_slides:
        selected_tiles_dir = normal_selected_tiles_dir if slide['label'] == 0 else tumor_selected_tiles_dir
        slide_tiles_coords = np.load(os.path.join(selected_tiles_dir, slide['slide_name'] + '.npy'))
        acc += len(slide_tiles_coords)

    return acc


def get_eval_ds_len(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir):
    slides_info = [*normal_slides_info, *tumor_slides_info]
    eval_slides = filter(lambda s: not s['slide_name'].startswith('b') and not s['slide_name'].startswith('f'), slides_info)

    acc = 0
    for slide in eval_slides:
        selected_tiles_dir = normal_selected_tiles_dir if slide['label'] == 0 else tumor_selected_tiles_dir
        slide_tiles_coords = np.load(os.path.join(selected_tiles_dir, slide['slide_name'] + '.npy'))
        acc += len(slide_tiles_coords)

    return acc


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
