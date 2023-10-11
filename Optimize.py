import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the dataset
dataset_name = 'chronic_kidney_disease.csv'
chronic_kidney_disease_dataframe = pd.read_csv(dataset_name)

# Preprocess the data
chronic_kidney_disease_dataframe = chronic_kidney_disease_dataframe.replace('?', np.nan)

knn_missing_values_imputer = KNNImputer(n_neighbors=5)
feature_classes = pd.DataFrame(knn_missing_values_imputer.fit_transform(chronic_kidney_disease_dataframe.iloc[:, 0:24]))

standard_feature_scaler = StandardScaler()
feature_classes = standard_feature_scaler.fit_transform(feature_classes)
feature_classes = pd.DataFrame(feature_classes, columns=chronic_kidney_disease_dataframe.columns[0:24])

target_label_encoder = LabelEncoder()
target_class = target_label_encoder.fit_transform(chronic_kidney_disease_dataframe['class'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_classes, target_class, test_size=0.3, random_state=42)

# Build a more complex neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with the Adam optimizer and a lower learning rate
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the model with more epochs
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predict on the test set
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Calculate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)

# Calculate accuracy and convert it to percentage format
accuracy = (confusion[0][0] + confusion[1][1]) / len(y_test) * 100

# Display classification report
class_report = classification_report(y_test, y_pred, target_names=['notckd', 'ckd'])

print('Confusion Matrix:\n', confusion)
print('Accuracy:', accuracy, '%')
print('Classification Report:\n', class_report)
