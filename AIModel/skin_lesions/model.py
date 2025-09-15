import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("ddi_metadata.csv")

le = LabelEncoder()
df['disease'] = le.fit_transform(df['disease'])

num_classes = df['disease'].nunique()

IMG_SIZE = (128,128)

def load_image(filename, label):
    img_path = os.path.join("ddidiversedermatologyimages")

    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0

    return img, label

filenames = df["DDI_file"].values
labels = df['disease'].values

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(load_image).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(dataset, epochs=5)

model.save("cnn_model_tf.h5")