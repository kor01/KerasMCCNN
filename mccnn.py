import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations


def preprocess_rgb(image):
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.image.rgb_to_grayscale(image)
  return tf.image.per_image_standardization(image)


def fast_mccnn_model(path=None, normalize=False):

  dirname = os.path.join(os.path.dirname(__file__), "weights")
  if os.path.exists(dirname):
    path = os.path.join(dirname, "mccnn")

  model = tf.keras.Sequential()
  model.add(layers.Lambda(preprocess_rgb))

  model.add(layers.Conv2D(
      64, (3, 3), padding='same', activation=activations.relu))
  model.add(layers.Conv2D(
    64, (3, 3), padding='same', activation=activations.relu))
  model.add(layers.Conv2D(
    64, (3, 3), padding='same', activation=activations.relu))
  model.add(layers.Conv2D(64, (3, 3), padding='same'))

  if normalize:
    model.add(layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=3)))

  if path is not None:
    model.load_weights(path)
  return model


def predict(model, image):

  if image.ndim == 3:
    batch, image = False, image[None]
  else:
    batch = True

  features = model.predict(image)

  if batch:
    return features

  return features[0, ...]


__all__ = ["predict", 'fast_mccnn_model']
