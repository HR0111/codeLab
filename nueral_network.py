import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Read the dataset
data = pd.read_csv('path/to/dataset.csv')  # Replace with the actual path to your dataset

# Step 2: Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Step 3: Distinguish feature and target set
X = data.drop('Exited', axis=1)  # Features
y = data['Exited']               # Target

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# Step 4: Divide the data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 5: Normalize the train and test data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Initialize and build the model
model = Sequential()
model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(Dense(8, activation='relu'))  # Hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Step 7: Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype(int)  # Predict class labels

# Calculate accuracy score and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("Confusion Matrix:\n", conf_matrix)



























