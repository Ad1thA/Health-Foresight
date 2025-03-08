import json
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")


# Loading diabetes dataset into a pandas DataFrame
diabetes_dataset = pd.read_csv('./diabetes.csv')

# Display the first 5 rows of the dataset
# print(diabetes_dataset.head())

# # Number of rows and columns in the dataset
# print(diabetes_dataset.shape)

# # Statistical summary of the dataset
# print(diabetes_dataset.describe())

# # Distribution of 'Outcome' values
# print(diabetes_dataset['Outcome'].value_counts())

# # Mean values grouped by 'Outcome'
# print(diabetes_dataset.groupby('Outcome').mean())

# Separating data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# print(X)
# print(Y)

# Standardizing the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# print(standardized_data)

X = standardized_data

# print(X)
# print(Y)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# print(X.shape, X_train.shape, X_test.shape)

# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# print('Accuracy score of the training data :', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# print('Accuracy score of the test data :', test_data_accuracy)

# Making a prediction
if __name__ == "__main__":
    no_preg = sys.argv[1]
    glucose = sys.argv[2]
    bl_pres = sys.argv[3]
    skin_thick = sys.argv[4]
    insulin = sys.argv[5]
    bmi = sys.argv[6]
    dia_func = sys.argv[7]
    age = sys.argv[8]

    # input_data = (4, 199, 70, 1, 4, 55.8, 0.553, 65)
    input_data = (no_preg, glucose, bl_pres, skin_thick, insulin, bmi, dia_func, age)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    # print(std_data)

    prediction = classifier.predict(std_data)
    # print(prediction)

    if prediction[0] == 0:
        # print('The person does not have diabetes')
        json_string = json.dumps({"prediction": "The person does not have diabetes"}, indent=4)  
        print(json_string)
    else:
        # print('The person has diabetes')
        json_string = json.dumps({"prediction": "The person has diabetes"}, indent=4)  
        
        print(json_string)

    # Save the trained model to a file
    with open('diabetes_model.pkl', 'wb') as file:
        pickle.dump(classifier, file)