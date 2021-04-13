import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from openslide.deepzoom import DeepZoomGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
<<<<<<< HEAD
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.layers as layers

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
=======
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from src.images import utils
from sklearn.metrics import classification_report
import random

def fine_tuning(normal_slides_info, tumor_slides_info, normal_selected_tiles_dir, tumor_selected_tiles_dir):


    #TODO: move to utils function
>>>>>>> 13588e77269a0aab549789857f1f8800b74cebca
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

<<<<<<< HEAD
    batch_size = 32
    train_gen = feed_slides_generator(train_slides_info, y_train, batch_size, mode="train")
    eval_gen = feed_slides_generator(test_slides_info, y_test, batch_size, mode="eval")
    test_gen = feed_slides_generator(test_slides_info, y_test, batch_size, mode="eval")

    train_len = get_ds_len(train_slides_info)
    val_len = get_ds_len(test_slides_info)

    base_model = ResNet50(weights='imagenet', include_top=True)

    x = base_model.output
=======
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
>>>>>>> 13588e77269a0aab549789857f1f8800b74cebca
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)

<<<<<<< HEAD
    model = Model(inputs=base_model.inputs, outputs=x)

    for layer in base_model.layers:
        layer.trainable = False

    print("[INFO] compiling model...")
    opt = SGD(learning_rate=1e-4, momentum=0.9)
=======
    model = Model(inputs=baseModel.inputs, outputs=x)

    #print(model.summary())

    for layer in baseModel.layers:
        layer.trainable = False

    print("[INFO] compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9)
>>>>>>> 13588e77269a0aab549789857f1f8800b74cebca
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    print("[INFO] training head...")
    H = model.fit(
        x=train_gen,
        steps_per_epoch=train_len // batch_size,
        validation_data=eval_gen,
        validation_steps=val_len // batch_size,
<<<<<<< HEAD
        epochs=10,
        shuffle=True
    )
=======
        epochs=10)
>>>>>>> 13588e77269a0aab549789857f1f8800b74cebca

    model.save("fine_tuned")

    predIdxs = model.predict(x=test_gen, steps=(val_len // batch_size) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)

    # print(classification_report({}, predIdxs))
    plot_training(H, 10)


<<<<<<< HEAD
def feed_slides_generator(slides_info, labels_info, batch_size, mode='train'):
    slide_num = 0
    data = np.array([]).reshape((0, 224, 224, 3))
    labels = []
    num_slides_prefetch = cfg.NUM_SLIDES_PREFETCH

    while True:
        tiles = []
        for i in range(num_slides_prefetch):
            current_slide = slides_info[slide_num+i]
            current_label = labels_info[slide_num+i]

=======
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
>>>>>>> 13588e77269a0aab549789857f1f8800b74cebca
            slide = utils.open_wsi(current_slide['slide_path'])
            zoom = DeepZoomGenerator(slide, tile_size=224, overlap=0)

            # Find the deep zoom level corresponding to the requested magnification
            dzg_level_x = utils.get_x_zoom_level(current_slide['highest_zoom_level'], current_slide['slide_magnification'],
                                                 10)
<<<<<<< HEAD
            selected_tiles_dir = cfg.selected_tiles_dir
            print(">> Getting tiles..")
            slide_tiles_coords = np.load(os.path.join(selected_tiles_dir, current_slide['slide_name'] + '.npy'))

            for coord in slide_tiles_coords:
                tile = zoom.get_tile(dzg_level_x, (coord[0], coord[1]))
                # noinspection PyBroadException
                try:
                    np_tile = utils.normalize_staining(tile)
                except IndexError:
                    print("Index error, skipping tile")
                    continue
                except:
                    print("Error in preprocessing, skipping tile")
                    continue
                tiles.append(np_tile)
                if mode == 'train' and current_label == cfg.NORMAL_LABEL:
                    for _ in range(cfg.MULTIPLIER):
                        # Facciamo attenzione a non introdurre alterazioni del colore
                        # (saturazione, contrasto, ecc) perché, essendo sbilanciati, rischierebbero
                        # di specializzare la rete neurale su queste alterazioni
                        data_augmentation = tf.keras.Sequential([
                            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                            layers.experimental.preprocessing.RandomRotation(0.2),
                            layers.experimental.preprocessing.RandomZoom(0.4),
                        ])
                        augmented_tile = data_augmentation(np.expand_dims(np_tile, axis=0))
                        tiles.append(np.squeeze(augmented_tile.numpy(), axis=0))

            data = np.concatenate((data, tiles))

            labels = [*labels, *(float(current_label) for s in range(len(tiles)))]
            print("Extracted ", len(slide_tiles_coords), " tiles")
            print("Now data is ", len(data), " and labels is ", len(labels))
            tiles = []

        print("Extracted ", num_slides_prefetch, " slides")
        print("The data len is ", len(data), " while batch_size is ", batch_size)

        # shuffling
        idx = np.random.permutation(len(data))
        data = data[idx]
        labels = np.array(labels)[idx]

        # batching
        while len(data) > batch_size:
            print("Yielding ", len(np.array(data[:batch_size])), len(np.array(labels[:batch_size]).reshape(-1, 1)))
            yield np.array(data[:batch_size]), np.array(labels[:batch_size]).reshape(-1, 1)
            data = np.delete(data, np.s_[:batch_size], 0)
            print("data is now ", len(data))
            labels = np.delete(labels, np.s_[:batch_size], 0)
            print("labels is now ", len(labels))

        slide_num += num_slides_prefetch
=======
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
>>>>>>> 13588e77269a0aab549789857f1f8800b74cebca
        if slide_num > len(slides_info):
            slide_num = 0


<<<<<<< HEAD
def get_ds_len(slides_info):
    acc = 0
    for slide in slides_info:
        coords_path = os.path.join(cfg.selected_tiles_dir, slide['slide_name'] + '.npy')
        slide_tiles_coords = np.load(coords_path)
        multiplier = cfg.MULTIPLIER if int(slide["label"]) == cfg.NORMAL_LABEL else 1  # Moltiplichiamo il numero di tile per il moltiplicatore per correggere la classe sbilanciata
        print("label is ", "normal" if int(slide["label"]) == cfg.NORMAL_LABEL else "tumor", "; then multiplier is ", multiplier)
        acc += multiplier * len(slide_tiles_coords)
=======
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
>>>>>>> 13588e77269a0aab549789857f1f8800b74cebca

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

<<<<<<< HEAD
# 1: separare test, val e train
# 2: capire come estrarre le tile, feedarle
# 3: froze / unfroze
# fare data augmentation
# unfroze di più livelli
# ottimizzare con asincrono / multithread
# cambiare metrica (matthew...): va in conflitto con SMOTE?
# 4: rileggere il capitolo prediction, e se facessimo il feed delle tile, poi facciamo la prediction di un'immagine dividendola in tile e si inferisce la soglia
# poi possiamo utilizzare lo stesso procedimento per etichettare le singole tile
=======

if __name__ == '__main__':
    fine_tuning()

# 1: separare test, val e train
# 2: capire come estrarre le tile, feedarle
# 3: froze / unfroze
# 4: rileggere il capitolo prediction, e se facessimo il feed delle tile, poi facciamo la prediction di un'immagine dividendola in tile e si inferisce la soglia
# poi possiamo utilizzare lo stesso procedimento per etichettare le singole tile
>>>>>>> 13588e77269a0aab549789857f1f8800b74cebca
