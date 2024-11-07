import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import datetime

def create_model():
    """Create a CNN model for skin cancer classification"""
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),  # Changed to use Input layer instead of input_shape
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    return model

def prepare_data_generators(data_dir):
    """Prepare data generators for training, validation, and testing"""
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for validation and testing
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def train_model(model, train_generator, validation_generator, epochs=20):
    """Train the model"""
    # Create callbacks
    checkpoint_path = os.path.join('models', 'best_model.keras')  # Changed file extension to .keras
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    return history

def evaluate_and_save_results(model, history, test_generator, results_dir):
    """Evaluate model and save results"""
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_history.png'))
    plt.close()

    # Generate predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

    # Save classification report
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

def main():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')

    # Create directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Prepare data
    train_generator, validation_generator, test_generator = prepare_data_generators(DATA_DIR)

    # Create and train model
    model = create_model()
    history = train_model(model, train_generator, validation_generator)

    # Evaluate and save results
    evaluate_and_save_results(model, history, test_generator, RESULTS_DIR)

if __name__ == "__main__":
    main()