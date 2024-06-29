import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model

def create_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_data(data_dir):
    # Data augmentation for the training set
    train_datagen = ImageDataGenerator(
    rescale=1.0/255,              # Rescale pixel values to [0, 1]
    rotation_range=40,            # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,        # Randomly translate images horizontally (fraction of total width)
    height_shift_range=0.2,       # Randomly translate images vertically (fraction of total height)
    shear_range=0.2,              # Randomly shear images
    zoom_range=0.2,               # Randomly zoom in/out on images
    horizontal_flip=True,         # Randomly flip images horizontally
    fill_mode='nearest',          # Fill in new pixels with the nearest pixel value
    brightness_range=[0.8, 1.2],  # Randomly change brightness (min 80%, max 120%)
    channel_shift_range=0.2,      # Randomly shift values in each channel
    contrast_stretching=True,     # Apply contrast stretching
    histogram_equalization=True,  # Apply histogram equalization
    adaptive_equalization=True    # Apply adaptive equalization
)


    # Only rescaling for the test set
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dataset = train_datagen.flow_from_directory(
        data_dir + '/train',
        target_size=(128, 128),  # Adjust image size based on your dataset
        batch_size=32,
        class_mode='sparse'
    )

    test_dataset = test_datagen.flow_from_directory(
        data_dir + '/test',
        target_size=(128, 128),  # Adjust image size based on your dataset
        batch_size=32,
        class_mode='sparse'
    )
    return train_dataset, test_dataset
