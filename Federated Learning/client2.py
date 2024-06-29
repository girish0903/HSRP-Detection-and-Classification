import flwr as fl
from model import create_model, load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class CustomDatasetClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_data, epochs=1)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data)
        return loss, len(self.test_data), {"loss": loss, "accuracy": accuracy}

def load_data(data_dir):
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_filenames = []
    train_labels = []

    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                train_filenames.append(os.path.join(root, file))
                label = os.path.basename(root)  # Assuming the folder name is the class label
                train_labels.append(label)

    # Split the filenames and labels into two parts
    train_filenames1, train_filenames2, train_labels1, train_labels2 = train_test_split(train_filenames, train_labels, test_size=0.5, stratify=train_labels)

    # Create separate generators for each part
    train_dataset1 = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_filenames1, 'class': train_labels1}),
        x_col='filename',
        y_col='class',
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse'
    )

    train_dataset2 = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_filenames2, 'class': train_labels2}),
        x_col='filename',
        y_col='class',
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse'
    )

    test_dataset = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse'
    )

    return train_dataset1, train_dataset2, test_dataset

def main():
    model = create_model()
    train_dataset1, train_dataset2, test_dataset = load_data("Classification.v1i.folder (1)")  # Specify the path to your custom dataset

    client = CustomDatasetClient(model, train_dataset2, test_dataset)
    
    fl.client.start_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
