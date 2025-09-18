import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import glob
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.regularizers import l2

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(42)
tf.random.set_seed(42)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        ce = -y_true * tf.math.log(y_pred)
        
        weight = alpha * y_true * tf.pow((1 - y_pred), gamma)
        
        fl = weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(fl, axis=1))
    
    return focal_loss_fixed

class MicroplasticClassifier:
    def __init__(self, img_size=(128, 128), batch_size=16):  
        tf.config.experimental.enable_op_determinism()
        
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        self.class_weights = None
        
    def create_stratified_split(self, data_dir, test_size=0.2, random_state=42):
        image_paths = []
        labels = []
        
        class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_dirs))}
        
        for class_name in class_dirs:
            class_path = os.path.join(data_dir, class_name)
            class_images = glob.glob(os.path.join(class_path, '*.jpg')) + \
                          glob.glob(os.path.join(class_path, '*.jpeg')) + \
                          glob.glob(os.path.join(class_path, '*.png'))
            
            image_paths.extend(class_images)
            labels.extend([class_to_idx[class_name]] * len(class_images))
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=test_size, 
            stratify=labels, random_state=random_state
        )
        
        return train_paths, val_paths, train_labels, val_labels, class_to_idx
        
    def load_and_preprocess_data(self, data_dir):
        train_paths, val_paths, train_labels, val_labels, class_to_idx = self.create_stratified_split(data_dir)
        
        unique_labels = np.unique(train_labels)
        class_weights_array = compute_class_weight(
            'balanced', 
            classes=unique_labels, 
            y=train_labels
        )
        self.class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        def preprocess_function(image):
            image = tf.image.rgb_to_grayscale(image)
            image = image / 255.0
            image = tf.image.grayscale_to_rgb(image)
            return image
        
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_function, 
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )      
        
        validation_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_function
        )
        
        temp_train_dir = 'temp_train'
        temp_val_dir = 'temp_val'
        
        self._create_temp_directories(train_paths, train_labels, temp_train_dir, class_to_idx)
        self._create_temp_directories(val_paths, val_labels, temp_val_dir, class_to_idx)
        
        train_generator = train_datagen.flow_from_directory(
            temp_train_dir,
            target_size=self.img_size,  
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            color_mode='rgb'  
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            temp_val_dir,
            target_size=self.img_size,  
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            color_mode='rgb'  
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Found {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        print(f"Classes: {self.class_names}")
        print(f"Class weights: {self.class_weights}")
        
        return train_generator, validation_generator
    
    def _create_temp_directories(self, paths, labels, temp_dir, class_to_idx):
        import shutil
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        for label in set(labels):
            class_name = idx_to_class[label]
            os.makedirs(os.path.join(temp_dir, class_name), exist_ok=True)
        
        for path, label in zip(paths, labels):
            class_name = idx_to_class[label]
            filename = os.path.basename(path)
            dest_path = os.path.join(temp_dir, class_name, filename)
            shutil.copy2(path, dest_path)
    
    def build_transfer_learning_model(self):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3) 
        )
        base_model.trainable = False
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax', dtype='float32')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001, clipnorm=1.0, clipvalue=0.5),
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_phase1(self, train_generator, validation_generator, epochs=20):
        print("Phase 1: Training top layers (base model frozen)")
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, 
                            verbose=1, cooldown=2),
            ModelCheckpoint('models/best_phase1_model.h5', monitor='val_accuracy', 
                          save_best_only=True, mode='max', verbose=1)
        ]
        
        steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
        validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)
        
        history1 = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        return history1
    
    def train_phase2(self, train_generator, validation_generator, epochs=30):
        print("Phase 2: Fine-tuning (unfreezing some layers)")
        
        self.model.layers[0].trainable = True
        for layer in self.model.layers[0].layers[:-20]:  
            layer.trainable = False
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.00001, clipnorm=1.0, clipvalue=0.5), 
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=7, min_lr=1e-8, 
                            verbose=1, cooldown=2),
            ModelCheckpoint('models/best_transfer_model.h5', monitor='val_accuracy', 
                          save_best_only=True, mode='max', verbose=1)
        ]
        
        steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
        validation_steps = max(1, validation_generator.samples // validation_generator.batch_size)
        
        history2 = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        return history2
    
    def train_model(self, train_generator, validation_generator, epochs=50):
        os.makedirs('models', exist_ok=True)
        
        phase1_epochs = min(20, epochs // 2)
        history1 = self.train_phase1(train_generator, validation_generator, phase1_epochs)
        
        phase2_epochs = epochs - phase1_epochs
        history2 = self.train_phase2(train_generator, validation_generator, phase2_epochs)
        
        self.history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        return self.history
    
    def evaluate_model(self, validation_generator):
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        validation_generator.reset()
        
        evaluation = self.model.evaluate(validation_generator, verbose=1)
        print(f"Validation Loss: {evaluation[0]:.4f}")
        print(f"Validation Accuracy: {evaluation[1]:.4f}")
        
        validation_generator.reset()
        predictions = self.model.predict(validation_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = validation_generator.classes
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
        
        self.plot_confusion_matrix(y_true, y_pred)
        
        return evaluation
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.show()
    
    def plot_training_history(self):
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(self.history['loss'], label='Training Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.show()
    
    def cleanup_temp_directories(self):
        import shutil
        
        temp_dirs = ['temp_train', 'temp_val']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

def main():
    print("Starting microplastic classification with Enhanced Transfer Learning...")
    
    classifier = MicroplasticClassifier(img_size=(128, 128), batch_size=16)
    
    data_dir = "dataset"
    
    try:
        print("Loading and preprocessing data with stratified splits...")
        train_gen, val_gen = classifier.load_and_preprocess_data(data_dir)
        
        print("Building enhanced transfer learning model...")
        model = classifier.build_transfer_learning_model()
        print(model.summary())
        
        print("Training model with enhanced transfer learning...")
        history = classifier.train_model(train_gen, val_gen, epochs=50)
        
        classifier.plot_training_history()
        
        print("Evaluating model...")
        evaluation = classifier.evaluate_model(val_gen)
        
    finally:
        classifier.cleanup_temp_directories()

if __name__ == "__main__":
    main()