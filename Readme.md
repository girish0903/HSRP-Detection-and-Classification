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

![Alt Text](path/to/image.png)


