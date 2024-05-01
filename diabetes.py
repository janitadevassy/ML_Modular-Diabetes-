import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

diabetes= pd.read_csv("C:\\Users\\janit\\Downloads\\archive (3)\\diabetes.csv")
print(diabetes.head())
print(diabetes.isnull().sum())


# Feature variables (all columns except the last one, which is the target)
X = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]  # Select features

# Target variable (the last column, indicating presence or absence of diabetes)
y = diabetes['Outcome']  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the training data (learn the mean and standard deviation)
scaler.fit(X_train)

# Transform both training and testing data using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a KNN classifier instance with k=5 neighbors (you can experiment with different k values)
knn_model = KNeighborsClassifier(n_neighbors=4)

# Train the model using the training data
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = knn_model.predict(X_test_scaled)

# Evaluate model performance using metrics (e.g., accuracy_score from scikit-learn)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)  # This will print the model's accuracy on the testing data

import pickle

# Save the trained model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

    