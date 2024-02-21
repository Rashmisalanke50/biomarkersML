#libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

# Load the training and testing data
train_data = pd.read_csv(r"C:\Users\rashm\OneDrive\Desktop\streamlit\Data\Project3\p1_train.csv")
test_data = pd.read_csv(r"C:\Users\rashm\OneDrive\Desktop\streamlit\Data\Project3\p1_test.csv")

# Rename columns for better understanding
train_data.columns = ['Feature1', 'Feature2', 'Label']
test_data.columns = ['Feature1', 'Feature2', 'Label']

# Check the shapes of the dataframes
print("Shape of train_data:", train_data.shape)
print("Shape of test_data:", test_data.shape)
# Display the first few rows of each dataframe
print("Train data:")
print(train_data.head())
print("\nTest data:")
print(test_data.head())

#check for null values
train_data.isna().sum()
test_data.isna().sum()

# Separate features and target variable for training and testing data
X_train = train_data[['Feature1', 'Feature2']]
y_train = train_data['Label']
X_test = test_data[['Feature1', 'Feature2']]
y_test = test_data['Label']

# Initialize the linear regression model
lr_model = LinearRegression()
# Train the model
lr_model.fit(X_train, y_train)
# Predict on the test data
lr_predictions = lr_model.predict(X_test)
# Calculate metrics
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
print("Linear Regression:")
print("Mean Squared Error:", lr_mse)
print("Mean Absolute Error:", lr_mae)

# Initialize the SVR model
svr_model = SVR()
# Train the model
svr_model.fit(X_train, y_train)
# Predict on the test data
svr_predictions = svr_model.predict(X_test)
# Calculate metrics
svr_mse = mean_squared_error(y_test, svr_predictions)
svr_mae = mean_absolute_error(y_test, svr_predictions)
print("Support Vector Regression (SVR):")
print("Mean Squared Error:", svr_mse)
print("Mean Absolute Error:", svr_mae)