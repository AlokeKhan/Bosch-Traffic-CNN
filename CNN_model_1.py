# === Imports ===
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN
import matplotlib.pyplot as plt
import os
import pickle




# === Paths (Update if needed) ===
train_dir = '/mnt/d/Traffic Dataset/processed_train_lights'
test_dir = '/mnt/d/Traffic Dataset/processed_test_lights'
model_save_path = '/mnt/d/Traffic Dataset/traffic_light_model_v2.h5'


# === Data Preparation ===
batch_size = 32
img_size = (32, 32)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
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

# === CNN Model ===
def make_model():
        
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
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
    return model
if __name__ == "__main__":

    model = make_model()
    # === Train Model ===
    print(f"Number of training samples: {train_generator.samples}")
    print(f"Number of test samples: {test_generator.samples}")
    print(f"Number of classes: {len(train_generator.class_indices)}")
    print(f"Class indices: {train_generator.class_indices}")
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=test_generator,
    
        verbose=1,            # Make sure verbose is 1 to see progress            # Reduce worker threads
    )
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\n Final Test Accuracy: {test_accuracy:.4f}")

    # === Save the Model ===
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")


# === Evaluate Model ===

# === Plot Training and Validation Accuracy ===
