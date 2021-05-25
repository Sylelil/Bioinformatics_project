import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from keras.callbacks import CSVLogger
from openslide.deepzoom import DeepZoomGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa

from config import paths
from src.images import utils
import config.images.config as cfg
import random


def fine_tuning(train_slides_info, test_slides_info, y_train, y_test):
    timer = utils.Time()
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

    train_gen = feed_slides_generator(train_slides_info, y_train, cfg.BATCH_SIZE, mode="train")
    eval_gen = feed_slides_generator(test_slides_info, y_test, cfg.BATCH_SIZE, mode="eval")
    test_gen = feed_slides_generator(test_slides_info, y_test, cfg.BATCH_SIZE, mode="eval")

    train_len = get_ds_len(train_slides_info)
    val_len = get_ds_len(test_slides_info)

    base_model = ResNet50(weights='imagenet', include_top=True)

    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)
,
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
    opt = SGD(learning_rate=1e-6, momentum=0.9)

    if os.path.exists(cfg.FINE_TUNED_MODEL_NAME):
        print(">> Loading model")
        model = tf.keras.models.load_model(cfg.FINE_TUNED_MODEL_NAME)
    else:
        print(">> Creating model")
        model.compile(loss="binary_crossentropy", optimizer=opt,
                    metrics=[metric, 'accuracy'])

    print("[INFO] training head...")

    csv_logger = CSVLogger("model_history_log.csv", append=True)

    history = model.fit(
        x=train_gen,
        steps_per_epoch=train_len // cfg.BATCH_SIZE,
        validation_data=eval_gen,
        validation_steps=val_len // cfg.BATCH_SIZE,
        epochs=cfg.NUM_EPOCHS,
        callbacks=[csv_logger],
        shuffle=True
    )

    model.save(cfg.FINE_TUNED_MODEL_NAME)

    plot_training(history, cfg.NUM_EPOCHS)

    print(">> Time to perform fine tuning: %s" % str(timer.elapsed()))

    #predIdxs = model.predict(x=test_gen, steps=(val_len // batch_size) + 1)
    #predIdxs = np.argmax(predIdxs, axis=1)

    #print(classification_report({}, predIdxs))


def feed_slides_generator(slides_info, y, batch_size, mode='train'):
    data = np.array([]).reshape((0, 224, 224, 3))
    labels = np.array([])
    files = list(map(lambda slide_info: slide_info["slide_name"].split("_")[0], slides_info))
    while True:
        for file in os.listdir(paths.selected_tiles_dir):
            if file.split("_")[1] not in files:
                continue
            file_path = os.path.join(paths.selected_tiles_dir, file)
            npy_img = np.load(file_path)

            current_label = 1 if file.endswith('0.npy') else 0
            npy_img = npy_img / 255.0

            data = np.append(data, np.expand_dims(npy_img, axis=0), axis=0)
            labels = np.append(labels, np.array(current_label), axis=None)

            if mode == 'train' and current_label == int(cfg.TUMOR_LABEL):
                for _ in range(cfg.MULTIPLIER):
                    # Facciamo attenzione a non introdurre alterazioni del colore
                    # (saturazione, contrasto, ecc) perché, essendo sbilanciati, rischierebbero
                    # di specializzare la rete neurale su queste alterazioni
                    data_augmentation = tf.keras.Sequential([
                        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                        layers.experimental.preprocessing.RandomRotation(0.2),
                        layers.experimental.preprocessing.RandomZoom(0.4),
                    ])

                    augmented_tile = data_augmentation(np.expand_dims(npy_img, axis=0))
                    augmented_tile_npy = augmented_tile.numpy()
                    data = np.append(data, augmented_tile_npy, axis=0)
                    labels = np.append(labels, np.array(current_label), axis=None)

            if len(data) > batch_size:
                yield np.array(data[:batch_size]), np.array(labels[:batch_size]).reshape(-1, 1)
                data = np.delete(data, np.s_[:batch_size], 0)
                labels = np.delete(labels, np.s_[:batch_size], 0)


def get_ds_len(slides_info):
    acc = 0
    for slide in slides_info:
        coords_path = os.path.join(paths.selected_coords_dir, slide['slide_name'] + '.npy')
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