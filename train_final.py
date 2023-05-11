import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the data
data = pd.read_csv("crop_data.csv")

# Split the data into features (X) and target (y)
X = data.drop("Crop", axis=1)
y = data["Crop"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier with the given hyperparameters
clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=1, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model
with open("random_forest_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Classification report:")
print(classification_report(y_test, y_pred))

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Overall classification accuracy:", accuracy)
