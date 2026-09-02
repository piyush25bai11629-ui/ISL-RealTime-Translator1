import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

def build_isl_model(sequence_length=30, num_features=225, num_classes=50):
    """
    Builds the CNN + LSTM architecture for ISL recognition.
    - sequence_length: 30 frames of video per sign
    - num_features: 225 (Pose: 99 + Left Hand: 63 + Right Hand: 63)
    - num_classes: 50 (based on the INCLUDE-50 dataset standards)
    """
    model = Sequential()
    
    # 1. CNN Layers (Feature Extraction)
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    
    # 2. LSTM Layers (Temporal/Sequence Learning)
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    
    # 3. Dense Classification Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

if __name__ == '__main__':
    # Test if the model compiles successfully and display architecture summary
    model = build_isl_model()
    model.summary()