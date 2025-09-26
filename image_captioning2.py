import os
import json
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Paths to data
train_images_path = "/media/e-soc-student/DISK2/Other_Project/wang_fashion_project/FACAD/TRAIN_IMAGES.hdf5"
train_captions_path = "/media/e-soc-student/DISK2/Other_Project/wang_fashion_project/FACAD/TRAIN_CAPTIONS.json"
val_images_path = "/media/e-soc-student/DISK2/Other_Project/wang_fashion_project/FACAD/VAL_IMAGES.hdf5"
val_captions_path = "/media/e-soc-student/DISK2/Other_Project/wang_fashion_project/FACAD/VAL_CAPTIONS.json"
word_map_path = "/media/e-soc-student/DISK2/Other_Project/wang_fashion_project/FACAD/WORDMAP.json"

# Load word map
with open(word_map_path, 'r') as f:
    word_map = json.load(f)
vocab_size = len(word_map)

# Load captions
def load_captions(captions_path):
    with open(captions_path, 'r') as f:
        captions = json.load(f)
    return captions

train_captions = load_captions(train_captions_path)
val_captions = load_captions(val_captions_path)

# Preprocessing function for images
def preprocess_image(img):
    # img = np.transpose(img, (1, 2, 0))  # Transpose from (channels, height, width) to (height, width, channels)
    img = tf.image.resize(img, (224, 224)).numpy()  # Resize to ResNet50 input size
    img = preprocess_input(img)  # Normalize
    return img

# Create a data generator for lazy loading
def load_images_generator(hdf5_path, batch_size):
    """
    Lazily load and preprocess images in batches from an HDF5 file.
    """
    with h5py.File(hdf5_path, 'r') as h:
        if 'images' not in h:
            raise KeyError("'images' key not found in the HDF5 file.")
        dataset = h['images']
        total_images = dataset.shape[0]
        print(f"Total images: {total_images}, Batch size: {batch_size}")
        
        for start in range(0, total_images, batch_size):
            end = min(start + batch_size, total_images)
            batch_images = np.array([preprocess_image(img) for img in dataset[start:end]])
            yield batch_images

# Extract features using the generator
def extract_features_with_generator(generator, model, batch_size):
    features = []
    for batch_images in generator:
        print(f"Processing batch with shape: {batch_images.shape}")
        batch_features = model.predict(batch_images, batch_size=batch_size)
        features.append(batch_features)
    return np.vstack(features)

# Feature extractor using ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model_extract_features = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features for training and validation datasets
batch_size = 32
train_images_gen = load_images_generator(train_images_path, batch_size)
train_features = extract_features_with_generator(train_images_gen, model_extract_features, batch_size)

val_images_gen = load_images_generator(val_images_path, batch_size)
val_features = extract_features_with_generator(val_images_gen, model_extract_features, batch_size)

# Preprocess captions
def create_sequences(tokenizer, captions, max_length, features):
    X1, X2, y = [], [], []
    for key, cap_list in captions.items():
        for cap in cap_list:
            seq = tokenizer.texts_to_sequences([cap])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(features[int(key)])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Tokenizer setup
tokenizer = Tokenizer()
tokenizer.fit_on_texts([item for sublist in train_captions.values() for item in sublist])
max_length = max(len(cap.split()) for cap_list in train_captions.values() for cap in cap_list)

# Generate training and validation data
X1_train, X2_train, y_train = create_sequences(tokenizer, train_captions, max_length, train_features)
X1_val, X2_val, y_val = create_sequences(tokenizer, val_captions, max_length, val_features)

# Define the model
inputs1 = Input(shape=(train_features.shape[1],))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = LSTM(256)(se1)

decoder1 = add([fe2, se2])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit([X1_train, X2_train], y_train, epochs=20, batch_size=64, validation_data=([X1_val, X2_val], y_val))

# Save the model
model.save("image_captioning_resnet50_model.h5")
