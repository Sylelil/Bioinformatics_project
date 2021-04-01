import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from openslide.deepzoom import DeepZoomGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from src.images import utils
import config.images.config as cfg
import random


def fine_tuning(train_slides_info, test_slides_info, y_train, y_test):

    seed = random.randint(0, 100)

    random.Random(seed).shuffle(train_slides_info)
    random.Random(seed).shuffle(test_slides_info)
    random.Random(seed).shuffle(y_train)
    random.Random(seed).shuffle(y_test)

    # TODO: move to utils function
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 500
    train_gen = feed_slides_generator(train_slides_info, y_train, batch_size, mode="train")
    eval_gen = feed_slides_generator(test_slides_info, y_test, batch_size, mode="eval")
    test_gen = feed_slides_generator(test_slides_info, y_test, batch_size, mode="eval")

    train_len = get_ds_len(train_slides_info)
    val_len = get_ds_len(test_slides_info)

    base_model = ResNet50(weights='imagenet', include_top=True)

    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

    model = Model(inputs=base_model.inputs, outputs=x)

    for layer in base_model.layers:
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


def feed_slides_generator(slides_info, labels_info, batch_size, mode='train'):
    slide_num = 0
    data = np.array([]).reshape((0, 224, 224, 3))
    labels = []
    num_tiles_prefetch = 3

    while True:
        tiles = []
        for i in range(num_tiles_prefetch):
            current_slide = slides_info[slide_num+i]
            current_label = labels_info[slide_num+i]

            slide = utils.open_wsi(current_slide['slide_path'])
            zoom = DeepZoomGenerator(slide, tile_size=224, overlap=0)

            # Find the deep zoom level corresponding to the requested magnification
            dzg_level_x = utils.get_x_zoom_level(current_slide['highest_zoom_level'], current_slide['slide_magnification'],
                                                 10)
            selected_tiles_dir = cfg.selected_tiles_dir
            print(">> Getting tiles..")
            slide_tiles_coords = np.load(os.path.join(selected_tiles_dir, current_slide['slide_name'] + '.npy'))

            for coord in slide_tiles_coords:
                tile = zoom.get_tile(dzg_level_x, (coord[0], coord[1]))
                np_tile = utils.normalize_staining(tile)
                tiles.append(np_tile)
            data = np.concatenate((data, tiles))

            labels = [*labels, *(float(current_label) for s in range(len(tiles)))]
            tiles = []
            print("Extracted ", len(slide_tiles_coords), " tiles")

        print("Extracted ", num_tiles_prefetch, " slides")
        print("The data len is ", len(data), " while batch_size is ", batch_size)
        while len(data) > batch_size:
            print("Yielding ", len(np.array(data[batch_size:])), len(np.array(labels[batch_size:]).reshape(-1, 1)))
            yield np.array(data[batch_size:]), np.array(labels[batch_size:]).reshape(-1, 1)
            np.delete(data, np.s_[batch_size:], 0)
            del labels[batch_size:]

        slide_num += num_tiles_prefetch
        if slide_num > len(slides_info):
            slide_num = 0


def get_ds_len(slides_info):
    acc = 0
    for slide in slides_info:
        coords_path = os.path.join(cfg.selected_tiles_dir, slide['slide_name'] + '.npy')
        slide_tiles_coords = np.load(coords_path)
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

# 1: separare test, val e train
# 2: capire come estrarre le tile, feedarle
# 3: froze / unfroze
# fare data augmentation
# cambiare metrica (matthew...): va in conflitto con SMOTE?
# 4: rileggere il capitolo prediction, e se facessimo il feed delle tile, poi facciamo la prediction di un'immagine dividendola in tile e si inferisce la soglia
# poi possiamo utilizzare lo stesso procedimento per etichettare le singole tile