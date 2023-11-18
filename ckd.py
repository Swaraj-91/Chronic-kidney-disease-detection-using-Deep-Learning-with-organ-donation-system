import joblib
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the saved model
loaded_model = load_model('ckd_model.keras')

# Load the standard scaler fitted with training data
standard_feature_scaler = joblib.load('standard_scaler.pkl')

# Define the attribute ranges for CKD detection
attribute_ranges = {
    'age': (0, 120),
    'bp': (60, 140),
    'sg': (1.003, 1.030),
    'al': (0, 30),
    'su': (0, 200),
    'pc': (0, 1),
    'pcc': (0, 1),
    'rbc': (4.0, 5.6),
    'ba': (3.5, 5.5),
    'bgr': (0, 199),
    'bu': (10, 40),
    'sc': (0.7, 1.3),
    'sod': (135, 150),
    'pot': (3.5, 5.0),
    'hemo': (3.1, 17.2),
    'pcv': (37, 50),
    'wbcc': (4000, 11000),
    'rbcc': (4.0, 6.0),
    'htn': (0, 1),
    'dm': (0, 1),
    'cad': (0, 1),
    'appet': (0, 1),
    'pe': (0, 1),
    'ane': (0, 1)
}

# Initialize CKD flag
ckd_flag = False

# Input features from the user
user_input = {}

for attribute, (min_value, max_value) in attribute_ranges.items():
    user_value = float(input(f"Enter the value for {attribute} (between {min_value} and {max_value}): "))
    
    # Implement conditions to detect CKD
    if (
        (attribute == 'bp' and (user_value < 60 or user_value > 140)) or
        (attribute == 'sg' and (user_value < 1.003 or user_value > 1.030)) or
        (attribute == 'al' and user_value > 30) or
        (attribute == 'su' and user_value > 200) or
        (attribute == 'rbc' and user_value < 4.0) or
        (attribute == 'ba' and user_value < 3.5) or
        (attribute == 'bgr' and user_value > 199) or
        (attribute == 'bu' and user_value > 40) or
        (attribute == 'sc' and user_value > 1.3) or
        (attribute == 'sod' and user_value > 150) or
        (attribute == 'pot' and  user_value > 5.0) or
        (attribute == 'hemo' and user_value < 3.1) or
        (attribute == 'pcv' and (user_value < 37 or user_value > 50)) or
        (attribute == 'htn' and user_value == 1) or
        (attribute == 'dm' and user_value == 1) or
        (attribute == 'cad' and user_value == 1) or
        (attribute == 'ane' and user_value == 1)
    ):
        ckd_flag = True

    user_input[attribute] = user_value

# Check if the patient has CKD or deficiency of attributes
if ckd_flag:
    print("The patient has chronic kidney disease.")
else:
    print("The patient does not have chronic kidney disease.")