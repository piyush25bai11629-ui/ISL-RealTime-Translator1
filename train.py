import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import build_isl_model

# Model & Data Configurations
SEQUENCE_LENGTH = 30
NUM_FEATURES = 225
NUM_CLASSES = 50
EPOCHS = 10
BATCH_SIZE = 16

def run_training_pipeline(data_path="../data/processed"):
    """
    Trains the CNN + LSTM model on extracted landmark sequences.
    """
    print("=== Starting Model Training Pipeline ===")
    
    X, y = [], []
    
    # Load dataset if processed numpy arrays exist
    if os.path.exists(data_path) and len(os.listdir(data_path)) > 0:
        print(f"Loading data from {data_path}...")
        for class_idx in range(NUM_CLASSES):
            class_folder = os.path.join(data_path, str(class_idx))
            if os.path.exists(class_folder):
                for file_name in os.listdir(class_folder):
                    if file_name.endswith('.npy'):
                        filepath = os.path.join(class_folder, file_name)
                        sequence = np.load(filepath)
                        X.append(sequence)
                        y.append(class_idx)
        X = np.array(X)
        y = np.array(y)
    else:
        print("No raw dataset found.")
        print("Generating synthetic sequence data to verify pipeline works...")
        X = np.random.rand(100, SEQUENCE_LENGTH, NUM_FEATURES).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, size=(100,))

    y_categorical = to_categorical(y, num_classes=NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42
    )

    # Build and Train model
    model = build_isl_model(
        sequence_length=SEQUENCE_LENGTH,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES
    )

    print("\nStarting training epochs...")
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Save trained model weights
    os.makedirs("../models", exist_ok=True)
    save_path = "../models/isl_model.h5"
    model.save(save_path)
    print(f"\nSUCCESS: Trained model saved to '{save_path}'")

if __name__ == "__main__":
    run_training_pipeline()