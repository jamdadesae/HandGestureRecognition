# **Hand Gesture Recognition System**

Welcome to the Hand Gesture Recognition System repository! 
This project is designed to recognize the alphabets of the English language using hand gestures. The system utilizes computer vision techniques and machine learning models to interpret hand gestures captured via a webcam or a video feed.

## **Overview**

The Hand Gesture Recognition System is capable of identifying English alphabets from hand gestures. This system can be used in various applications, including:

- **Sign Language Translation**: Enabling real-time translation of sign language into text, thereby facilitating communication for the deaf and hard-of-hearing community.
- **Human-Computer Interaction**: Allowing users to interact with computers and other digital devices through natural hand gestures, enhancing accessibility and user experience.
- **Assistive Technologies**: Providing tools for individuals with disabilities to communicate more effectively and access information through intuitive hand gestures.
- **Educational Tools**: Assisting in the teaching and learning of sign language and improving literacy skills through interactive applications.
- **Entertainment and Gaming**: Offering innovative ways to control and interact with games and virtual environments using hand gestures.

## **Features**

- **Real-time Hand Gesture Recognition**: Recognizes English alphabets in real-time using a webcam.
- **Pre-trained Model**: Comes with a pre-trained model that can be used out-of-the-box.
- **Custom Model Training**: Allows users to train their own models with custom datasets.
- **User-friendly Interface**: Easy-to-use graphical interface for capturing and recognizing gestures.

## **Usage**

**Data Collection**
The data collection script is designed to capture images of hand gestures. These images are then used to train the model to recognize different hand gestures corresponding to English alphabets. The script leverages two powerful libraries: OpenCV and CvZone.

OpenCV is an open-source computer vision and machine learning software library. It contains a comprehensive set of functions and tools for real-time image and video processing.
CvZone is a computer vision library that simplifies the use of OpenCV for tasks like hand detection and gesture recognition. It provides higher-level abstractions for common operations, making it easier to develop computer vision applications.

To run the data collection code, ensure your webcam is connected. Execute the script to start capturing images of hand gestures. Position your hand to form the desired gesture, and press the 's' key to save each image. Captured images will be stored in the specified folder (Data/Z) for further training. Repeat the process to collect a sufficient number of images for accurate model training. Save the Collected data in Data folder in your project file.

**Testing the Model**
The test script uses the pre-trained model to recognize hand gestures in real-time and display the predicted alphabet. 

**Model Training**
To train your own model, follow these steps:

- **Collect Data**: Use the data collection script to capture images of hand gestures. Ensure you capture a diverse set of images for each alphabet to improve model accuracy.
- **Preprocess Data**: Ensure all images are of uniform size and properly labeled. Use image augmentation techniques to increase the dataset size and variety.
- **Train Model**: Use a deep learning framework such as TensorFlow or PyTorch to train the model on your dataset. You can use convolutional neural networks (CNNs) for image classification tasks.
- **Save Model**: Save the trained model as keras_model.h5 and the corresponding labels in labels.txt in your Project file.


## **Acknowledgements**
This project uses the following open-source libraries:

- **OpenCV**: An open-source computer vision and machine learning software library.
- **CvZone**: A computer vision library that simplifies image processing and gesture detection.
- **NumPy**: A fundamental package for scientific computing with Python.
- **TensorFlow**: An end-to-end open-source platform for machine learning.
- **Keras**: A high-level neural networks API, written in Python and capable of running on top of TensorFlow.

## **Note**
I have excuted the entire code in pycharm environment.

## **Output**

![image](https://github.com/jamdadesae/HandGestureRecognition/assets/168914412/3960b350-3f26-45dd-8899-35c785aafc45)

![image](https://github.com/jamdadesae/HandGestureRecognition/assets/168914412/adf44eb4-8702-43ec-bfa1-751f79439c1b)

![image](https://github.com/jamdadesae/HandGestureRecognition/assets/168914412/a341e722-9563-44c1-a87c-395ff2e794f7)

![image](https://github.com/jamdadesae/HandGestureRecognition/assets/168914412/30b6bbc8-c316-48e8-905f-e6b8f53eb9de)




