# complete deep learning model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
dataset_name = 'chronic_kidney_disease.csv'
chronic_kidney_disease_dataframe = pd.read_csv(dataset_name)

# Define a function to convert non-numeric columns to numeric
def convert_to_numeric(column):
    try:
        return pd.to_numeric(column, errors='coerce')
    except ValueError:
        return column

# Apply the conversion to all columns
chronic_kidney_disease_dataframe = chronic_kidney_disease_dataframe.apply(convert_to_numeric)

# Preprocess the data
chronic_kidney_disease_dataframe = chronic_kidney_disease_dataframe.replace('?', np.nan)

# Filter the dataset to include only rows where the target class is 1
positive_class_data = chronic_kidney_disease_dataframe[chronic_kidney_disease_dataframe['class'] == 1]

# Calculate and display the range of each attribute for the positive class data
for column in positive_class_data.columns:
    if column != 'class':
        min_value = positive_class_data[column].min()
        max_value = positive_class_data[column].max()
        attribute_range = max_value - min_value
        print(f"Attribute: {column}, Range: ({min_value}, {max_value}), Range Length: {attribute_range}")

# Split the dataset into features and target class
X = chronic_kidney_disease_dataframe.drop(columns=['class'])
y = LabelEncoder().fit_transform(chronic_kidney_disease_dataframe['class'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with the Adam optimizer
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Print accuracy and confusion matrix
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(confusion)

# Save the trained model
model.save('my_model.h5')

# Plot the confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names):
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))

    # Create heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    # Add labels, title, and ticks
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

# Define class names (adjust as needed)
class_names = ['Not CKD', 'CKD']

# Plot the confusion matrix
plot_confusion_matrix(confusion, class_names)
plt.show()

# Print a message to indicate the completion of model training
print("Model training is complete.")
