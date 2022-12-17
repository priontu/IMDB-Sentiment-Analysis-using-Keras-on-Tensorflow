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
import pandas as pd

pd.set_option("display.colheader_justify","left")

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

def preprocess_and_predict(test_data, defined_eval_size = 50):
    test_size = defined_eval_size
    datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
    vocabulary = Counter()
    for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
        for review in X_batch:
            vocabulary.update(list(review.numpy()))

    vocab_size = 10000
    truncated_vocabulary = [
        word for word, count in vocabulary.most_common()[:vocab_size]]

    word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
    # for word in b"This movie was faaaaaantastic".split():
    #     print(word_to_id.get(word) or vocab_size)

    words = tf.constant(truncated_vocabulary)
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    num_oov_buckets = 4000
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)  

    def encode_words(X_batch, y_batch):
        return table.lookup(X_batch), y_batch

    test_pipe = test_data.batch(test_size).map(preprocess).map(encode_words).prefetch(1)

    model = keras.models.load_model('IMDB_Sentiment_Analysis_Model.h5')

    for feats, labs in test_pipe.unbatch().batch(test_size).take(1):
    probPreds = model.predict(feats)
    print(probPreds)

    for review, label in test_data.batch(test_size).take(1):
    pass

    positive_review_predicted_probability = probPreds
    negative_review_predicted_probability = 1 - probPreds

    # print(review)
    # print(label)
    # print(list(tf.transpose(positive_review_predicted_probability)))
    # print(negative_review_predicted_probability)

    review_df = pd.Series(review, dtype = object)
    review_df.name = "Reviews"
    label_df = pd.Series(label)
    label_df.name = "Original Label"
    pos_probs_df = pd.Series(list(tf.transpose(positive_review_predicted_probability))[0])
    pos_probs_df.name = "Positive_Predicted_Probability"
    neg_probs_df = pd.Series(list(tf.transpose(negative_review_predicted_probability))[0])
    neg_probs_df.name = "Negative_Predicted_Probability"
    res = pd.concat([review_df, label_df, pos_probs_df, neg_probs_df], axis = 1)

    print(res)

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
# print(datasets)
# X_train, y_train = datasets["train"]
# X_test, y_test = datasets["test"]

# X_test = np.concatenate([x for x, y in datasets['test'].batch(10).take(1)], axis=0)


# print(info.splits['unsupervised'].num_examples)

preprocess_and_predict(datasets['test'], defined_eval_size = 20)


