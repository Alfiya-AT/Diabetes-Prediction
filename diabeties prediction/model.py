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


# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# # import pickle
# # pickle.dump(RF, open("model.pkl", "wb"))

# import warnings
# warnings.filterwarnings('ignore')

# data = pd.read_csv("diabetes.csv")

# data.head()

# data.tail()

# data.shape

# data.dtypes

# data.info()

# data.describe().T

# df = data.copy(deep = True)

# print(df.isin([0]).sum())

# df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# print(df.isnull().sum())

# df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)

# df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)

# df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace = True)

# df['Insulin'].fillna(df['Insulin'].mean(), inplace = True)

# df['BMI'].fillna(df['BMI'].mean(), inplace = True)

# print(df.isnull().sum())

# df.head()

# print(df['Outcome'].value_counts())
# sns.countplot(df['Outcome'])

# df.corr()

# plt.figure(figsize=(10,10))
# sns.heatmap(df.corr())

# x=df[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction", "Age"]]
# y=df["Outcome"]

# x.head()

# y.head()

# from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
# print(" x\t\t y ".center(30))
# print ('Train set:', x_train.shape,  y_train.shape)
# print ('Test set:', x_test.shape,  y_test.shape)

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score

# LR = LogisticRegression()
# LR.fit(x_train, y_train)

# y_pred_LR = LR.predict(x_test)
# print("Test set Accuracy: ",accuracy_score(y_test, y_pred_LR))

# RF = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0,
#                                 max_features = 'auto', max_depth = 10)
# RF.fit(x_train, y_train)

# y_pred_RF = RF.predict(x_test)
# print("Test set Accuracy: ",accuracy_score(y_test, y_pred_RF))

# DTC=DecisionTreeClassifier()
# DTC.fit(x_train,y_train)

# y_pred_DTC = DTC.predict(x_test)
# print("Test set Accuracy: ",accuracy_score(y_test, y_pred_DTC))

# from sklearn import metrics

# Ks = 10
# mean_acc = np.zeros((Ks-1))

# for n in range(1,Ks): #1-10
    
#     #Train Model and Predict  
#     neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train,y_train)
#     yhat=neigh.predict(x_test)
#     mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

# mean_acc

# print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

# KNC = KNeighborsClassifier(n_neighbors = 1)
# KNC.fit(x_train,y_train)

# y_pred_KNC=KNC.predict(x_test)
# print("Test set Accuracy: ",accuracy_score(y_test, y_pred_KNC))

# print("Logistic Regression :",accuracy_score(y_test, y_pred_LR))
# print("Random Forest Classifier :",accuracy_score(y_test, y_pred_RF))
# print("Decision Tree Classifier :",accuracy_score(y_test, y_pred_DTC))
# print("K-Nearest Neighbors Classifier:",accuracy_score(y_test, y_pred_KNC))
