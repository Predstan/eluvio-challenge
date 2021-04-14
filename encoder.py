import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


class SentenceEncode(Model):
  def __init__(self):
    super(SentenceEncode, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(128, activation="relu", input_shape= (512,)), 
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),]) # Encoded Layer Defined Here
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(512, activation="sigmoid"),]) # Output mapped to reproduce data
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
