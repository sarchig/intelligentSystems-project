# Intelligent Systems Project

## Project Title
Traffic Sign Classification with Context-Aware Scene Description using Deep Learning

## Overview
This project investigates traffic sign classification and context-aware scene description using deep learning techniques. Traffic signs are first extracted from annotated street-level images and classified using convolutional neural network backbones. The predicted sign-level information is then combined with global scene understanding to generate human-readable, context-aware descriptions of driving environments. The proposed pipeline emphasizes interpretability by integrating localized traffic sign recognition with holistic scene-level reasoning, and its effectiveness is demonstrated through comparative evaluation of multiple CNN models and an end-to-end inference example.

---

## Repository Structure
intelligentSystems-project/
├── outputs/ # Generated figures and results
├── train_keras_signs.py # Model training script
├── inspect_mtsd.py # Dataset inspection and visualization
├── infer_one_test_image_blip.py # Single-image inference demo
├── save_crops_csv.py # Cropped image metadata generation and save
├── compare_select_models.py # Model comparison and selection
└── README.md # Project documentation


---

## How to Run the Project
1. Download the dataset using the link provided below.
2. Place the dataset in the `data/` directory.
3. Open and run all in `inspect_mtsd.py, save_crops_csv.py, train_keras_signs.py, compare_select_models.py, infer_one_test_image_blip.py` sequentially.
4. Generated results and figures will be saved in the `outputs/` directory.


---

## Dataset
Mapillary, “Mapillary traffic sign dataset,” https://www.mapillary.com/dataset/trafficsign, 2023


