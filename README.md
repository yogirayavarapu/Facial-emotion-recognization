# Facial Emotion Recognition
===========================

## Project Overview
-------------------

This project is a facial emotion recognition system that uses computer vision and deep learning to identify emotions from facial expressions. The system can recognize five emotions: happy, sad, neutral, surprise, and angry.

## Technologies Used
--------------------

* **Python**
* **OpenCV** (for face detection using Haarcascades)
* **Keras** (for building and training the CNN model)
* **TensorFlow** (backend for Keras)

## Dataset
----------

The dataset used for training the model is from Kaggle: [Facial Expression Recognition (FERChallenge)](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge)

## Model Performance
-------------------

The trained model achieves an accuracy of **75%** on the test dataset.

## Usage
-----

### Option 1: Train the model from scratch

1. Clone the repository
2. Install the required dependencies (listed in `requirements.txt`)
3. Run training.ipynb notebook code on Kaggle and save the model
4. Download the trained models and save on same directory
5. Run `python detection.py` to use the model for facial emotion recognition

### Option 2: Use the pre-trained model

1. Clone the repository
2. Download the pre-trained model from [Kaggle](https://www.kaggle.com/code/nandakishore230/facial-emotion-recognition/output)
3. Run `python detection.py` to use the model for facial emotion recognition

## Detection Script
-------------------

The `detection.py` script uses the OpenCV library to capture video from the webcam and detect faces using Haarcascades. The detected faces are then passed through the trained CNN model to recognize the emotion.


## Contact

If you have any questions or feedback, please feel free to reach out.

Thank you for using Facial Emotion Recognition! 😊
