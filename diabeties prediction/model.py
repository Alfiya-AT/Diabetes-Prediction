# STEP 1: Import necessary libraries
import pandas as pd
import numpy as np
import pickle  # to save model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# STEP 2: Load the dataset
data = pd.read_csv("diabetes.csv")

# STEP 3: Replace 0 with NaN in specific columns (since 0 is invalid in these)
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)

# STEP 4: Fill missing values (NaN) with mean of the column
for col in cols_with_zero:
    data[col] = data[col].fillna(data[col].mean())


# STEP 5: Define features (X) and target (y)
X = data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
          "BMI", "DiabetesPedigreeFunction", "Age"]]  # 8 features
y = data["Outcome"]  # Target

# STEP 6: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 7: Create and train the Random Forest model
RF = RandomForestClassifier(
    n_estimators=100,        # number of trees in the forest
    criterion='entropy',     # how to measure quality of split
    random_state=0,          # for reproducibility
    max_features='sqrt',     # use all features when looking for best split
    max_depth=10             # depth of each tree
)

RF.fit(X_train, y_train)  # Train the model on training data

# STEP 8: Check the model accuracy
y_pred = RF.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# STEP 9: Save the trained model to a file using pickle
pickle.dump(RF, open("model.pkl", "wb"))  # wb = write binary
print("Model saved successfully as model.pkl")

