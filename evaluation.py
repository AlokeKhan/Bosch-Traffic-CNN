import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import get_path
from CNN_model_1 import model_save_path
import pickle 

with open('history.pkl', 'rb') as f:
    history_data = pickle.load(f)

# Plot Accuracy
plt.figure()
plt.plot(history_data['accuracy'], label='Train Acc')
plt.plot(history_data['val_accuracy'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_accuracy.png')
plt.close()

# Plot Loss
plt.figure()
plt.plot(history_data['loss'], label='Train Loss')
plt.plot(history_data['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
plt.close()

def evaluate_model(model, test_generator):
    """
    Evaluate the model and generate confusion matrix and classification report
    """
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Get class labels
    class_labels = list(test_generator.class_indices.keys())
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusionmatrix.png")
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

def main():
    # Load the trained model
    model = load_model(model_save_path)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        '/mnt/d/Traffic Dataset/processed_test_lights',
        target_size=(32, 32),
        batch_size=32,
        class_mode='categorical',
        shuffle=False 
    )
    
    # Evaluate the model
    evaluate_model(model, test_generator)

if __name__ == "__main__":
    main() 