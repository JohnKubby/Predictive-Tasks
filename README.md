# Predictive-Tasks
# Predicting Customer Gender based on some shopping behaiour and other relevant features

#  Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
Dataset_path= "C:/Users/Laptop/Downloads/Customer data trends.csv"

df = pd.read_csv(Dataset_path)
df.info()
#  Basic EDA
print(df.info())
print(df.describe())
print(df.isnull().sum()) # Check for missing values

# droping the duplicate columns.
df_cleaned = df.drop_duplicates(keep=False)
df

# ploting the histogram.

df.hist(figsize=(10,8))
plt.show()



#  Encode the target variable 'Gender' because it's categorical
# Assuming 'Gender' has values like 'Male', 'Female', etc.
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Encoding 'Gender'


for column in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])


# Display the first few rows of the processed dataset
print(df.head())

#  Visualizing relationships 
# Heatmap to see correlations
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# Pairplot for visualizing relationships
sns.pairplot(df)
plt.show()

sns.pairplot(df, hue="Gender")  
plt.show()

# You can use more visualizations based on your columns
# For example, plot distributions of numeric features
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    
    
    
    
    
colors = sns.color_palette("husl", n_colors=len(df.select_dtypes(include=np.number).columns))

# Loop through each numerical column and plot

gender_colors = {"Male": "blue", "Female": "pink"}
for i, col in enumerate(df.select_dtypes(include=np.number).columns):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, color=colors[i])
    plt.title(f'Distribution of {col}')
    plt.show()


# Separate features (X) and target (y)
X = df.drop('Gender', axis=1)  # Dropping target column
y = df['Gender'] # target variable


# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train.info()
y_train

X_test.info()
y_test

# distrubution of target variables 



plt.figure(figsize = (6,4))
sns.countplot(df = df , X ='gender', palette = 'set2')
plt.title('Distrubution of gender')
plt.xlabel('Gender')
plt.ylabel('count')
plt.show

#sns.countplot(df=df, x='Season')


plt.figure(figsize = (8,5))
sns.boxplot(df, x = 'Gender', y = 'Purchase Amount (USD)')
plt.title('purchase Amount by Gender')
plt.xlabel('Gender')
plt.ylable('Purchase Amount (USD)')
plt.show

plt.figure(figsize = (8,5))
sns.boxplot(df, x = 'Gender', y = 'Review Rating')
plt.title('review rating by gender')
plt.xlabel('Gender')
plt.ylabel('review rating')
plt.show

plt.figure(figsize = (8,5))
sns.boxplot(df, x = 'Gender', y = 'Previous Purchases')
plt.title('review rating by gender')
plt.xlabel('Gender')
plt.ylabel('Previous Purchases')
plt.show

# initialize the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
###########################################################


# training  and building the models.

# support vector machine

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred=svc.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",svc.score(X_train,y_train)*100)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

#Precision: Both classes (0 and 1) have very high precision scores (close to 1.0), meaning the model rarely misclassifies either class.
#Recall: The recall is similarly high, indicating the model effectively captures almost all actual instances of each class.
#F1-Score: This score, which balances precision and recall, is nearly 1.0 for both classes, demonstrating excellent model performance across both accuracy and robustness in predictions.
#Confusion Matrix: The model made very few misclassifications. Specifically:
#Class 0: 228 instances predicted correctly, 0 misclassified.
#Class 1: 552 instances predicted correctly, only 3 misclassified.
#Training Score (Accuracy): The training accuracy is approximately 99.8%, which is very high and suggests the model has learned the patterns in the data very well.



## Random forest model

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",rfc.score(X_train,y_train)*100)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


#Classification Report:

#Precision, Recall, F1-Score: All scores are at 1.00 for both classes, which means:
#Precision of 1.00: Every prediction made for each class is correct, with no false positives.
#Recall of 1.00: The model captures all instances of each class, with no false negatives.
#F1-Score of 1.00: A balance of precision and recall, showing perfect predictive performance.
#Accuracy: The overall accuracy is 1.00 (100%), indicating that the model correctly predicted every single instance in the test set.

#Confusion Matrix:

#The confusion matrix confirms that the model has no misclassifications, with all instances correctly predicted as either class 0 or class 1.
#Training Score: A perfect 100% on the training score shows that the model has fully learned the patterns in the training data.


# DTC
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')

dtree.fit(X_train,y_train)


y_pred=dtree.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Training Score:\n",dtree.score(X_train,y_train)*100)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")


# Getting feature importances
feature_names = ['Age', 'Item Purchased', 'Category', 'Purchase Amount (USD)', 'Location', 'Size', 'Color ', 'Season', 'Review Rating', 'Subscription Status', 'Payment Method', 'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Previous Purchases', 'Preferred Payment Method','Frequency of Purchases']  # Replace with your actual feature names

# Get the column names from the DataFrame created by pd.get_dummies()
encoded_feature_names = df[column].columns
print(encoded_feature_names)

# Get feature importances from the trained Decision Tree model
feature_importances = dtree.feature_importances_

# Create a DataFrame to pair encoded feature names with their importances
feature_importance_df = pd.DataFrame({
    'Feature': encoded_feature_names,
    'Importance': feature_importances
})

# Sort by importance. Selecting most relevant features
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(feature_importance_df)


# Select the top 10 features
top_10_features = feature_importance_df.head(10)
print("Top 10 Features:\n", top_10_features)

from sklearn.neighbors import KNeighborsClassifier


# K-Nearest Neighbors, using the scaled X train
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train , y_train)

y_pred_knn = knn.predict(X_test)
conf_matrix_KNN = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix:\n", conf_matrix_KNN)
print("KNN Model Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn, target_names=['Male', 'Female']))
print("Training Score:\n",knn.score(X_train,y_train)*100)
