# Data Science Projects

This repository contains several data science projects:

## Sentiment Analysis using NLTK and RoBERTa

In this project, sentiment analysis was performed on a dataset containing reviews of Amazon products. The goal was to compare the results of a simple sentiment analysis using the NLTK library with the results of a pretrained RoBERTa model.

- nltk has the ability to perform sentiment analysis. It will provide a rating of whether a stamement is positive, negative or neutral;
- RoBERTa is a pretrained model used for sentimental analysis.

Link to dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## Fake or Real News Detection using Gensim and spaCy

spaCy and Gensim are open-source Python libraries used in Natural Language Processing (NLP). This project involved training machine learning models on natural language text using these libraries to detect fake or real news.

## Restaurant Chatbot

A basic chatbot for a restaurant was built using Dialogflow. The chatbot allows users to make orders, edit orders by adding or removing items, and inquire about their order. The backend API was built using FastAPI, and MySQL database was used to store order information. The frontend consists of a simple HTML page that displays the chatbot.

## Tomato Leaf Disease Classification using Convolutional Neural Network (CNN)

This project involves classifying a dataset of tomato leaves with different diseases into one of five classes using a CNN built in TensorFlow.

Link to data: https://www.kaggle.com/datasets/arjuntejaswi/plant-village

### Exploring the Data

- The data was imported into a Google Colab notebook from Google Drive.
- Some imbalance was found in the data, with some classes having just a thousand images while others have over 3000. Training was initially done with the data as is, with potential modifications to be made if the results are not appropriate.
- The quality of the images was considered acceptable after printing out the leaves to inspect them.

### Data Split and Preprocessing

- The data was split into training, test, and validation datasets.
- The images were resized and scaled, which is important for neural networks.
- Image augmentation was performed on the images to provide the model with different versions of the images.

### Model Training

- The model was given three convolutional layers, each with a 3x3 pixel size.
- Each convolutional layer was followed by a 2x2 max pooling layer and a ReLU activation function.
- The image matrices were flattened and fed into a neural network.
- The network had a single hidden layer to start.
- The model was trained for 100 epochs.

### Current Results and Observations

- The accuracy score for the model is 96%, but the precision, recall, and F1 scores are low, indicating overfitting.
- In the next iteration, a dropout rate will be added, and regularization will be applied.
- The number of epochs might also be reduced to around 50, as the validation loss was lowest around this point.

### Frontend and API

- A simple API has been created using FastAPI.
- A simple frontend using ReactJS will be added to make this an end-to-end project.

