import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the trained model
model = load_model('my_model.h5')  # Replace with the path to your trained model file

# Create an empty dictionary to collect user input
user_input = {}

# Collect input from the user for 24 attributes
for i in range(1, 25):
    attribute_name = f'attribute{i}'
    value = input(f'Enter the value for {attribute_name}: ')
    user_input[attribute_name] = [float(value)]

# Create a DataFrame from user input
new_input_data = pd.DataFrame(user_input)

# Preprocess the input data (similar to how you preprocessed the training data)
new_input_data = new_input_data.replace('?', np.nan)

knn_missing_values_imputer = KNNImputer(n_neighbors=5)
new_input_data = pd.DataFrame(knn_missing_values_imputer.fit_transform(new_input_data))

standard_feature_scaler = StandardScaler()
new_input_data = standard_feature_scaler.transform(new_input_data)

# Make a prediction
prediction = model.predict(new_input_data)

if prediction > 0.5:
    result = "Chronic Kidney Disease (CKD)"
else:
    result = "No Chronic Kidney Disease"

print("Prediction:", result)
