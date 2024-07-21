# Import necessary libraries and modules
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import openslide
from PIL import Image

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths to datasets
train_dir = '/kaggle/input/bach-breast-cancer-histology-images/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/Photos'
test_dir = '/kaggle/input/bach-breast-cancer-histology-images/ICIAR2018_BACH_Challenge_TestDataset/ICIAR2018_BACH_Challenge_TestDataset/Photos'
train_wsi_dir = '/kaggle/input/bach-breast-cancer-histology-images/ICIAR2018_BACH_Challenge/ICIAR2018_BACH_Challenge/WSI'
test_wsi_dir = '/kaggle/input/bach-breast-cancer-histology-images/ICIAR2018_BACH_Challenge_TestDataset/ICIAR2018_BACH_Challenge_TestDataset/WSI'

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

def extract_patches(slide, level=0, patch_size=IMG_SIZE, max_patches=100):
    width, height = slide.level_dimensions[level]
    patches = []
    for _ in range(max_patches):
        x = np.random.randint(0, width - patch_size)
        y = np.random.randint(0, height - patch_size)
        patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
        patches.append(np.array(patch))
    return patches

def process_wsi_files(wsi_dir, patch_size=IMG_SIZE, max_patches_per_slide=100):
    wsi_data = []
    wsi_labels = []
    
    for class_name in os.listdir(wsi_dir):
        class_path = os.path.join(wsi_dir, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith('.svs'):
                    wsi_path = os.path.join(class_path, file_name)
                    try:
                        slide = openslide.OpenSlide(wsi_path)
                        patches = extract_patches(slide, patch_size=patch_size, max_patches=max_patches_per_slide)
                        wsi_data.extend(patches)
                        wsi_labels.extend([class_name] * len(patches))
                    except Exception as e:
                        print(f"Error processing {file_name}: {str(e)}")
    
    return np.array(wsi_data), np.array(wsi_labels)

# Process WSI files
train_wsi_data, train_wsi_labels = process_wsi_files(train_wsi_dir)
test_wsi_data, test_wsi_labels = process_wsi_files(test_wsi_dir)

# Normalize WSI data
train_wsi_data = train_wsi_data.astype('float32') / 255.0
test_wsi_data = test_wsi_data.astype('float32') / 255.0

# Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Combine WSI data with regular image data
train_data = []
train_labels = []

for _ in range(len(train_generator)):
    batch = next(train_generator)
    train_data.append(batch[0])
    train_labels.append(batch[1])

train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)

if len(train_wsi_data) > 0:
    train_data = np.concatenate([train_data, train_wsi_data])
    train_labels = np.concatenate([
        train_labels,
        tf.keras.utils.to_categorical(
            [train_generator.class_indices[label] for label in train_wsi_labels],
            num_classes=train_generator.num_classes
        )
    ])

# Split data for ensemble training
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Define and compile models
def create_model(base_model, name):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

efficientnet_model = create_model(EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)), 'EfficientNet')
resnet_model = create_model(ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)), 'ResNet')
densenet_model = create_model(DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)), 'DenseNet')

# Callbacks
checkpoint = ModelCheckpoint('best_model_{epoch:02d}.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)

# Train models
models = [efficientnet_model, resnet_model, densenet_model]
histories = []

for model in models:
    print(f"Training {model.name}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop]
    )
    histories.append(history)

# Extract features for ML models
def extract_features(model, data):
    feature_extractor = Model(inputs=model.inputs, outputs=model.layers[-3].output)
    return feature_extractor.predict(data)

X_train_features = np.concatenate([extract_features(model, X_train) for model in models], axis=1)
X_val_features = np.concatenate([extract_features(model, X_val) for model in models], axis=1)

# Train ML models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

ml_models = [rf_model, gb_model, svm_model]
for model in ml_models:
    model.fit(X_train_features, np.argmax(y_train, axis=1))

# Ensemble prediction function
def ensemble_predict(models, ml_models, data):
    dl_predictions = np.mean([model.predict(data) for model in models], axis=0)
    features = np.concatenate([extract_features(model, data) for model in models], axis=1)
    ml_predictions = np.mean([model.predict_proba(features) for model in ml_models], axis=0)
    return (dl_predictions + ml_predictions) / 2

# Evaluate on validation set
val_predictions = ensemble_predict(models, ml_models, X_val)
val_accuracy = accuracy_score(np.argmax(y_val, axis=1), np.argmax(val_predictions, axis=1))
print(f"Validation Accuracy: {val_accuracy}")

# Prepare test data
test_images = []
test_image_filenames = []

for filename in os.listdir(test_dir):
    if filename.endswith(".tif"):
        img_path = os.path.join(test_dir, filename)
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        test_images.append(img_array)
        test_image_filenames.append(filename)

if test_images:
    test_images = np.vstack(test_images)
    
    # Predict classes for test images
    test_predictions = ensemble_predict(models, ml_models, test_images)
    predicted_classes = np.argmax(test_predictions, axis=1)
    
    # Map class indices to class names
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}
    
    # Print predictions
    for i, filename in enumerate(test_image_filenames):
                print(f"Image: {filename}, Predicted class: {class_labels[predicted_classes[i]]}")

# Define the ensemble model
inputs = tf.keras.Input(shape=(...))  # Define the input shape
x = models[0](inputs)  # Initialize with the first model
for model in models[1:]:
    x = model(x)  # Connect the output of each model to the next
ensemble_model = tf.keras.Model(inputs=inputs, outputs=x)  # Create the ensemble model

# Save the model
tf.saved_model.save(ensemble_model, 'ensemble_model')



