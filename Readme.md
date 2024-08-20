# High Security Registration Plate Detection and Classification

This project presents a novel approach for the detection and classification of High Security Registration Plates (HSRP) on vehicles. The proposed method utilizes object detection to locate HSRPs in images, followed by a classification step to determine if the detected object is an HSRP or not. The system aims to improve vehicle monitoring and public safety while maintaining data privacy.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [YOLOv8 for Number Plate Detection](#yolov8-for-number-plate-detection)
  - [Extracting Detected Number Plates](#extracting-detected-number-plates)
  - [Federated Learning for Classification](#federated-learning-for-classification)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Introduction

Detecting vehicles with High Security Registration Plates (HSRP) is vital for ensuring security and regulatory compliance on roadways. Traditional HSRP detection methods often lack efficiency and struggle to keep up with evolving vehicle registration systems. By leveraging Federated Learning, a privacy-preserving machine learning approach, this project presents a promising solution to this challenge.

## Dataset
We collected a diverse dataset of vehicle images from various sources. Each image was annotated to identify the presence and location of HSRPs. This annotation process was crucial for training our object detection model.

![Dataset](https://github.com/girish0903/HSRP-Detection-and-Classification/blob/main/Picture1.jpg)
*Annotated dataset showing vehicles with High Security Registration Plates*

## Methodology
### YOLOv8 for Number Plate Detection

We used YOLOv8 (You Only Look Once version 8) to detect number plates in the images. YOLOv8 is a state-of-the-art object detection model that balances speed and accuracy. The model was trained on our annotated dataset to accurately locate number plates in various conditions.

### Extracting Detected Number Plates

After detecting the number plates, we extracted these regions from the images. These extracted number plate images were then used as input for the classification phase.

### Federated Learning for Classification

The extracted number plate images were passed into a Federated Learning setup for classification. Federated Learning allows multiple decentralized devices to collaboratively train a model while keeping the data localized. This approach helps in maintaining data privacy. Our Federated Learning setup used MobileNet v3 as the model for classifying whether a detected number plate is an HSRP or not.

## Results

The results demonstrate the effectiveness of our approach:

The YOLOv8 model achieved an mAP(Mean Average Precision) of 0.993 in detecting number plates.
The Federated Learning-based classification achieved an accuracy of 91.66% in identifying HSRPs.

### Object Detection 

![yolo](https://github.com/girish0903/HSRP-Detection-and-Classification/blob/main/Picture2.jpg)

