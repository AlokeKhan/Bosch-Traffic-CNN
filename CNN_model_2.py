# === Imports ===
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# === Paths ===
train_dir = '/mnt/d/Traffic Dataset/processed_train_lights'
test_dir = '/mnt/d/Traffic Dataset/processed_test_lights'

# === Data Preparation ===
batch_size = 32
img_size = (32, 32)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.6, 1.4],
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# === CNN Model (Upgraded) ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),  # Extra Conv Layer
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: Red, Yellow, Green
])

# === Compile Model ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Add EarlyStopping Callback ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === Train Model ===
history = model.fit(
    train_generator,
    epochs=30,  # Train longer
    validation_data=test_generator,
    callbacks=[early_stop]
)

# === Evaluate Model ===
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n Final Test Accuracy: {test_accuracy:.4f}")

# === Save the Model ===
model_save_path = '/mnt/d/Traffic Dataset/traffic_light_model_v2.h5'
model.save(model_save_path)
print(f" Model saved at {model_save_path}")

# === Plot Training and Validation Accuracy ===
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# === Plot Training and Validation Loss ===
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
