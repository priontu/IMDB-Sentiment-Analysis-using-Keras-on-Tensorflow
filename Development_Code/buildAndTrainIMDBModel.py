# from google.colab import drive
# drive.mount('/content/gdrive')

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# # Is this notebook running on Colab or Kaggle?
# IS_COLAB = "google.colab" in sys.modules
# IS_KAGGLE = "kaggle_secrets" in sys.modules

# if IS_COLAB:
#     %pip install -q -U tensorflow-addons
#     %pip install -q -U transformers

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# if not tf.config.list_physical_devices('GPU'):
#     print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
#     if IS_COLAB:
#         print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")
#     if IS_KAGGLE:
#         print("Go to Settings > Accelerator and select GPU.")

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import tensorflow_datasets as tfds
from collections import Counter

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "nlp"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, rb" ", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b""), y_batch


datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
print(datasets)

for X_batch, y_batch in datasets["train"].batch(4).take(1):
    for review, label in zip(X_batch.numpy(), y_batch.numpy()):
        print("Review:", review.decode("utf-8")[:200], "...")
        print("Label:", label, "= Positive" if label else "= Negative")
        print()

# preprocess(X_batch, y_batch)


vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))


vocab_size = 12000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]




word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
for word in b"This movie was faaaaaantastic".split():
    print(word_to_id.get(word) or vocab_size)



words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 2000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)  



def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets["train"].batch(32).map(preprocess).map(encode_words).prefetch(1)
valid_set = datasets["test"].batch(32).map(preprocess).map(encode_words).prefetch(1)


for X_batch, y_batch in train_set.take(1):
    print(X_batch)
    print(y_batch)


embed_size = unit_size = 128
ss = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-1,
    decay_steps=10000,
    decay_rate=0.7)
# ss=5e-1
optimizer = keras.optimizers.SGD(learning_rate=ss)

checkpoint_cb=keras.callbacks.ModelCheckpoint('IMDB_Sentiment_Analysis_Model.h5', 
                                              save_best_only=True)
earlyStop_cb=keras.callbacks.EarlyStopping(patience=10,
                                           restore_best_weights=True)
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, 
                           embed_size,
                           mask_zero=True, # not shown in the book
                           input_shape=[None]),
    keras.layers.GRU(unit_size, 
                     return_sequences=True),
    # keras.layers.GRU(unit_size, 
    #                  return_sequences=True),
    keras.layers.GRU(unit_size),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", 
              optimizer=optimizer, 
              metrics=["accuracy"])

history = model.fit(train_set, 
                    epochs=30, 
                    # steps_per_epoch = train_size//32,
                    # batch_size = unit_size, 
                    validation_data = valid_set, 
                    callbacks=[checkpoint_cb, earlyStop_cb])
