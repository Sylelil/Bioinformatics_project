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
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

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

    batch_size = 32
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

    # non effettuiamo il freeze dell'ultimo strato convoluzionale
    for layer in base_model.layers[-cfg.NUM_TRAINABLE_LAYERS:]:
        layer.trainable = True

    for layer in base_model.layers:
        print("{}: {}".format(layer.name, layer.trainable))

    metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2, name='MatthewsCorrelationCoefficient')
    print("[INFO] compiling model...")
    opt = SGD(learning_rate=1e-4, momentum=0.9)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=[metric, 'accuracy'])

    print("[INFO] training head...")
    history = model.fit(
        x=train_gen,
        steps_per_epoch=train_len // batch_size,
        validation_data=eval_gen,
        validation_steps=val_len // batch_size,
        epochs=1,
        shuffle=True
    )

    model.save(cfg.FINE_TUNED_MODEL_NAME)

    plot_training(history, 10)

    #predIdxs = model.predict(x=test_gen, steps=(val_len // batch_size) + 1)
    #predIdxs = np.argmax(predIdxs, axis=1)

    #print(classification_report({}, predIdxs))


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
                # noinspection PyBroadException
                try:
                    np_tile = utils.normalize_staining(tile)
                except IndexError:
                    print("IndexError, skipping tile")
                    continue
                except:
                    print("Unknown error, skipping tile")
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

                        # TODO: controllo per evitare OOM. Avrebbe senso toglierlo probabilmente in quanto limita il data augmentation
                        if len(tiles) > 3500:
                            break
                        augmented_tile = data_augmentation(np.expand_dims(np_tile, axis=0))
                        tiles.append(np.squeeze(augmented_tile.numpy(), axis=0))

            data = np.concatenate((data, tiles))

            labels = [*labels, *(-1 if float(current_label) == 0 else 1 for s in range(len(tiles)))]
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
        if slide_num > len(slides_info):
            slide_num = 0


def get_ds_len(slides_info):
    acc = 0
    for slide in slides_info:
        coords_path = os.path.join(cfg.selected_tiles_dir, slide['slide_name'] + '.npy')
        slide_tiles_coords = np.load(coords_path)
        multiplier = (1+cfg.MULTIPLIER) if int(slide["label"]) == cfg.NORMAL_LABEL else 1  # Moltiplichiamo il numero di tile per il moltiplicatore per correggere la classe sbilanciata
        print("label is ", "normal" if int(slide["label"]) == cfg.NORMAL_LABEL else "tumor", "; then multiplier is ", multiplier)
        acc += multiplier * len(slide_tiles_coords)

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
# unfroze di più livelli
# ottimizzare con asincrono / multithread
# cambiare metrica (matthew...): va in conflitto con SMOTE?
# 4: rileggere il capitolo prediction, e se facessimo il feed delle tile, poi facciamo la prediction di un'immagine dividendola in tile e si inferisce la soglia
# poi possiamo utilizzare lo stesso procedimento per etichettare le singole tile