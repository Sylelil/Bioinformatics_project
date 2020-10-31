from contextlib import nullcontext
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import openslide
import openslide.deepzoom


data_dir = Path('..') / 'datasets' / 'images'
# data_dir = 'D:\\Bioinformatics_project\\datasets'

class_names = {
  'normal': 0,
  'tumor': 1
}
train_ratio = 0.8
def load_images():
  list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*/*.svs'))
  list_ds_count = len(list_ds)
  print("Count is {}".format(list_ds_count))
  print("First is {}".format(list_ds.take(1)))
  val_size = int(list_ds_count * train_ratio)
  train_ds = list_ds.take(val_size)
  val_ds = list_ds.skip(val_size)
  train_ds = train_ds.map(process_path)
  val_ds = val_ds.map(process_path)
  for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

def get_label(file):
  print(file)
  return ""
  class_str = str(file.numpy()).split(os.path.sep)[-5]
  class_name = class_names[class_str]
  return class_name

def process_path(file_path):
  label = get_label(file_path)
  img = convert_openslide(file_path)
  return img, label
    

def convert_openslide(file):
  print(file)
  return file
  slide = openslide.OpenSlide(str(file))
  img = slide.get_thumbnail((512, 512))
  return tf.keras.preprocessing.image.img_to_array(img)  


if __name__ == '__main__':
  load_images()