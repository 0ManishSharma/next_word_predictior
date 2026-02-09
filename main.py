import tensorflow as tf

path = tf.keras.utils.get_file("shakespeare.txt",
                               "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")
text = open(path, "rb").read().decode("utf-8")
print(len(text), "characters loaded")
