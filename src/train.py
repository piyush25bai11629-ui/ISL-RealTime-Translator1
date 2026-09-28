import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_isl_model

# Model & Data Configurations
SEQUENCE_LENGTH = 30
NUM_FEATURES = 225
NUM_CLASSES = 50
EPOCHS = 12
BATCH_SIZE = 16

# Class label mapping for ISL gestures 45-49
LABEL_MAPPING = {
    45: "I Love You",
    46: "Yes",
    47: "No",
    48: "Namaste",
    49: "Thank You"
}

def run_training_pipeline(data_path="data/processed", epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Trains the CNN + LSTM model on extracted landmark sequences,
    evaluates on validation set, and generates performance plots.
    """
    print("=== Starting Model Training Pipeline ===")
    
    # Resolve relative data_path if needed
    if not os.path.exists(data_path) and os.path.exists(os.path.join("..", data_path)):
        data_path = os.path.join("..", data_path)
    
    X, y = [], []
    
    # Load dataset if processed numpy arrays exist
    if os.path.exists(data_path) and len(os.listdir(data_path)) > 0:
        print(f"Loading data from {data_path}...")
        for class_idx in range(NUM_CLASSES):
            class_folder = os.path.join(data_path, str(class_idx))
            if os.path.exists(class_folder):
                label_name = LABEL_MAPPING.get(class_idx, f"Class {class_idx}")
                files = [f for f in os.listdir(class_folder) if f.endswith('.npy')]
                print(f"  Found class {class_idx} ({label_name}): {len(files)} sequences")
                for file_name in files:
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

    print(f"Loaded dataset: X shape = {X.shape}, y shape = {y.shape}")

    y_categorical = to_categorical(y, num_classes=NUM_CLASSES)

    # Stratified 80/20 train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples | Val set: {X_test.shape[0]} samples")

    # Build model (architecture 100% unchanged)
    model = build_isl_model(
        sequence_length=SEQUENCE_LENGTH,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES
    )

    # Model checkpointing for saving the best model
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_isl_model.keras")
    
    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_categorical_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    print("\nStarting training epochs...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Save final model
    save_path = os.path.join(models_dir, "isl_model.h5")
    model.save(save_path)
    print(f"\nSUCCESS: Trained final model saved to '{save_path}'")
    print(f"SUCCESS: Best model checkpoint saved to '{best_model_path}'")

    # Save label mapping
    label_map_path = os.path.join(models_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in LABEL_MAPPING.items()}, f, indent=4)
    print(f"SUCCESS: Label mapping saved to '{label_map_path}'")

    # Load best model for evaluation
    best_model = tf.keras.models.load_model(best_model_path)
    print("\n=== Evaluating Best Checkpoint on Validation Set ===")
    val_loss, val_acc = best_model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
    print(f"Best Model - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
    print(f"Best Model - Val Loss:   {val_loss:.4f}, Val Accuracy:   {val_acc*100:.2f}%")

    # Predictions & Confusion Matrix
    y_pred_probs = best_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    eval_classes = sorted(list(LABEL_MAPPING.keys()))
    class_names = [f"{LABEL_MAPPING[c]} ({c})" for c in eval_classes]

    cm = confusion_matrix(y_true, y_pred, labels=eval_classes)
    print("\n=== Confusion Matrix (Classes 45-49) ===")
    print("Labels order:", class_names)
    print(cm)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, labels=eval_classes, target_names=class_names, zero_division=0))

    # Plot & Save Training Loss and Accuracy Curves
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
    plt.title('Training and Validation Loss', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Crossentropy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['categorical_accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_categorical_accuracy'], label='Val Accuracy', marker='s')
    plt.title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    curves_path = os.path.join(models_dir, "loss_accuracy_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"SUCCESS: Loss and accuracy curves saved to '{curves_path}'")

    # Plot & Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title('Confusion Matrix - ISL Classes 45-49', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"SUCCESS: Confusion matrix plot saved to '{cm_path}'")

    # Save summary stats to json
    summary_data = {
        "final_epoch_metrics": {
            "train_loss": float(history.history['loss'][-1]),
            "train_accuracy": float(history.history['categorical_accuracy'][-1]),
            "val_loss": float(history.history['val_loss'][-1]),
            "val_accuracy": float(history.history['val_categorical_accuracy'][-1]),
        },
        "best_checkpoint_metrics": {
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc)
        },
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }
    summary_path = os.path.join(models_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"SUCCESS: Training summary saved to '{summary_path}'")

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL Model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--data-path", type=str, default="data/processed", help="Path to processed data")
    args = parser.parse_args()

    run_training_pipeline(data_path=args.data_path, epochs=args.epochs, batch_size=args.batch_size)