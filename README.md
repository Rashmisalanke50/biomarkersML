# biomarkersML
This code performs regression on biological sensor data to predict an organism's health. It utilizes linear regression and support vector regression models to predict health metrics and evaluates using mean squared error and mean absolute error.
Health Prediction from Biological Sensor Data
Overview
This repository contains code for predicting an organism's health based on measurements from two biological sensors. It implements linear regression and support vector regression models to predict health metrics and evaluates the model performance using mean squared error and mean absolute error.

Problem Statement
The task is to predict the current health of an organism given measurements from two biological sensors. The dataset includes features representing bio-markers, and the target variable is the organism's health.

Dataset
The dataset is divided into training and testing sets. The training set (p1_train.csv) is used to train the models, while the testing set (p1_test.csv) is used to evaluate model performance.

Models Used
Linear Regression: A basic regression model that fits a linear relationship between features and target variable.
Support Vector Regression (SVR): A regression model that uses support vector machines to find the best-fitting line.
Usage
Clone the repository:
git clone https://github.com/username/repository.git
Install dependencies:
pip install -r requirements.txt
Load the training and testing data from the provided CSV files.

Train the linear regression and SVR models on the training data.

Evaluate model performance using mean squared error and mean absolute error on the testing data.

Compare the performance of both models and select the best-performing one for health prediction.

Results
The code provides insights into the performance of linear regression and SVR models for predicting organism health based on biological sensor data. The evaluation metrics help in assessing the effectiveness of each model.
