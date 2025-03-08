import json
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

"""Data Collection and Processing"""

# Load CSV to pandas DataFrame
heart_data = pd.read_csv('./heart_disease_data.csv')

# Print the first 5 rows
# print(heart_data.head())

# # Print the last 5 rows
# print(heart_data.tail())

# # Number of rows and columns
# print(heart_data.shape)

# # Info about the data
# print(heart_data.info())

# # Checking for missing values
# print(heart_data.isnull().sum())

# # Statistical measures
# print(heart_data.describe())

# # Checking the distribution of the target variable
# print(heart_data['target'].value_counts())

"""1 == Having Heart Disease
0 == Not having Heart Disease
"""

# Splitting the features and the target (1 or 0)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# print(X)
# print(Y)

"""Splitting the data into Training data and Test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Checking the number of training data and test data
# print(X.shape, X_train.shape, X_test.shape)

"""Machine Learning Model Training - Logistic Regression"""

# Increase max_iter and specify solver to address convergence issues
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Training the LR model with training data
model.fit(X_train, Y_train)

"""Model Evaluation - Accuracy Score"""

# Accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# print('Accuracy on training data:', training_data_accuracy)

# Accuracy on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# print('Accuracy on test data:', test_data_accuracy)

"""Predictive system"""
if __name__ == "__main__":
    age = int(sys.argv[1])
    sex = int(sys.argv[2])
    chestPain = int(sys.argv[3])
    restingBloodPressure = int(sys.argv[4])
    serumCholesterol = int(sys.argv[5])
    fastingBloodSugar = int(sys.argv[6])
    restingECG = int(sys.argv[7])
    maxHeartRate = int(sys.argv[8])
    exerciseAngina = int(sys.argv[9])
    stDepression = float(sys.argv[10])
    stSlope = int(sys.argv[11])
    majorVessels = int(sys.argv[12])
    thal = int(sys.argv[13])

    # print(age,sex,chestPain,restingBloodPressure,serumCholesterol,fastingBloodSugar,restingECG,maxHeartRate,exerciseAngina,stDepression,stSlope,majorVessels,thal)

    # input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
    input_data = (age,sex,chestPain,restingBloodPressure,serumCholesterol,fastingBloodSugar,restingECG,maxHeartRate,exerciseAngina,stDepression,stSlope,majorVessels,thal)

    #changing data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    # print(prediction)

    if (prediction[0] == 0):
      # print('The person does not have Heart Disease')
      json_string = json.dumps({"prediction": "The person does not have Heart Disease"}, indent=4)  
      print(json_string)
    else:
      # print('The person has Heart Disease')
      json_string = json.dumps({"prediction": "The person has Heart Disease"}, indent=4)  
      print(json_string)

    with open('heart_disease_model.pkl', 'wb') as file:
        pickle.dump(model, file)