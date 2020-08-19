import tensorflow as tf
from sentence_transformers import SentenceTransformer
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  if len(gpus)>=2:
    tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[2], True)

txt_samples = ["firma name", "G", "A",  "kaltes bier", "vogel", "bird", "pelikan", "imperceptron", "albatros coding", "tree leaves", "nice coffee", "german engineering", "abcdef ghij", "klmnopq", "rstu vwxyz", "0123456789"]

with tf.device('/CPU:0'):
  bert = SentenceTransformer("distiluse-base-multilingual-cased")
  t = bert.encode(np.array(txt_samples))
