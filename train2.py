# Complete Deep learning Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset
dataset_name = 'chronic_kidney_disease.csv'
chronic_kidney_disease_dataframe = pd.read_csv(dataset_name)

# Filter rows where 'class' is not equal to 0
filtered_dataset = chronic_kidney_disease_dataframe[chronic_kidney_disease_dataframe['class'] != 0]

# Assuming 'age' is the column you want to delete
columns_to_drop = ['appet', 'pe', 'class', 'age', 'pc', 'pcc', 'wbcc', 'rbcc']
filtered_dataset = filtered_dataset.drop(columns=columns_to_drop, axis=1)

# Assuming you want to calculate the range for multiple attributes
selected_attributes = ['bp','sg', 'al', 'su', 'rbc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'htn', 'dm', 'cad', 'ane']

# Convert selected attributes to numeric (if needed)
filtered_dataset[selected_attributes] = filtered_dataset[selected_attributes].apply(pd.to_numeric, errors='coerce')

# Get the range for each selected attribute excluding NaN values
attribute_ranges = filtered_dataset[selected_attributes].agg(['min', 'max'])

# Display the range of specific attributes
print("Range of selected attributes:")
print(attribute_ranges)


# Data Preprocessing
def preprocess_data(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])
    return df, scaler, label_encoder

chronic_kidney_disease_dataframe, scaler, label_encoder = preprocess_data(chronic_kidney_disease_dataframe)

X = chronic_kidney_disease_dataframe.drop('class', axis=1)
y = chronic_kidney_disease_dataframe['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Learning Rate Scheduler
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

_, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

y_pred = (model.predict(X_test) > 0.5).astype("int32")
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

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

class_names = ['Not CKD', 'CKD']
plot_confusion_matrix(confusion, class_names)
plt.show()

model.save('ckd_model.keras')
joblib.dump(scaler, 'standard_scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model training is complete.")