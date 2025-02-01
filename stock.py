import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from skopt import gp_minimize
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, MultiHeadAttention
import pandas as pd
df=pd.read_csv("C:\\Users\\Thanu\OneDrive\\Documents\\Desktop\\yahoofinance.csv")
df.head()  # Display the first few rows
df.shape   # Get the dimensions of the data
df.describe()  # Summary statistics
df.info()


# LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, MultiHeadAttention
from sklearn.metrics import mean_squared_error
import numpy as np


# Convert date to timestamp
df=pd.read_csv("C:\\Users\\Thanu\OneDrive\\Documents\\Desktop\\yahoofinance.csv")
df['ds'] = pd.to_datetime(df['ds'])
df['ds'] = df['ds'].astype('int64') / 10**9
print(df.dtypes)

# Normalize features (excluding 'Adj Close**'column)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['Adj Close**']))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Adj Close**'], test_size=0.2, random_state=42)

# Create an LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print("Adj Close**")
print(f"Train Score (RMSE): {train_score:.2f}")
print(f"Test Score (RMSE): {test_score:.2f}")

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['Close*']))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Close*'], test_size=0.2, random_state=42)

# Create an LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print("Close*")
print(f"Train Score (RMSE): {train_score:.2f}")
print(f"Test Score (RMSE): {test_score:.2f}")
y_pred = regressorGRU.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['Low']))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Low'], test_size=0.2, random_state=42)

# Create an LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=10, validation_data=(X_test, y_test))

# Evaluate the model
train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print("Low")
print(f"Train Score (RMSE): {train_score:.2f}")
print(f"Test Score (RMSE): {test_score:.2f}")


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['High']))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['High'], test_size=0.2, random_state=42)

# Create an LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print("High")
print(f"Train Score (RMSE): {train_score:.2f}")
print(f"Test Score (RMSE): {test_score:.2f}")


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['Volume']))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Volume'], test_size=0.2, random_state=42)

# Create an LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print("Volume")
print(f"Train Score (RMSE): {train_score:.2f}")
print(f"Test Score (RMSE): {test_score:.2f}")


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['Open']))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Open'], test_size=0.2, random_state=42)

# Create an LSTM model
model = Sequential()
model.add(LSTM(units=64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
train_score = model.evaluate(X_train, y_train, verbose=0)
test_score = model.evaluate(X_test, y_test, verbose=0)
print("Open")
print(f"Train Score (RMSE): {train_score:.2f}")
print(f"Test Score (RMSE): {test_score:.2f}")


# ACTUAL PRICE VS PREDICTION PRICE USING LSTM
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
valid_data = pd.read_csv("C:\\Users\\Thanu\OneDrive\\Documents\\Desktop\\yahoofinance.csv")
valid_data['ds'] = pd.to_datetime(valid_data['ds'])
valid_data['ds'] = valid_data['ds'].astype('int64') / 10**9
print(valid_data.dtypes)
dates = valid_data['ds']
X = valid_data.drop(['Volume'], axis=1)
y = valid_data['Close*']
actual_prices = valid_data['Close*']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the LSTM model
regressorLSTM = Sequential()
regressorLSTM.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressorLSTM.add(Dropout(0.2))
regressorLSTM.add(LSTM(units=50, return_sequences=True))
regressorLSTM.add(Dropout(0.2))
regressorLSTM.add(LSTM(units=50, return_sequences=True))
regressorLSTM.add(Dropout(0.2))
regressorLSTM.add(LSTM(units=50))
regressorLSTM.add(Dropout(0.2))
regressorLSTM.add(Dense(units=1))

regressorLSTM.compile(optimizer='adam',loss='mean_squared_error')
regressorLSTM.fit(X_train,y_train,epochs=50,batch_size=32)

# Make predictions using the test set
predictions_changes = regressorLSTM.predict(X_test)

# Convert the changes back to actual values
predictions_df = pd.DataFrame({'ds': dates[len(predictions_changes):], 'y': actual_prices[-len(predictions_changes):] + np.cumsum(predictions_changes)})

# Plot the actual and predicted prices
plt.figure(figsize=(10, 6))
plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
plt.plot(dates[:len(predictions_df)], predictions_df, label='Predicted Prices', color='orange')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.grid(True)
plt.show()

# GRADIENT BOOSTING
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Read data from the CSV file (replace with your actual file path)
file_path = "C:\\Users\\Thanu\\OneDrive\\Documents\\Desktop\\yahoofinance.csv"
df = pd.read_csv("C:\\Users\\Thanu\\OneDrive\\Documents\\Desktop\\yahoofinance.csv")

# Convert the "ds" column to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Extract timestamps (in seconds) from the datetime column
df['ds'] = df['ds'].astype('int64') // 10**9

# Assuming 'Low' is the target variable (classification problem)
X = df.drop(columns=['Low'])  # Features (excluding 'Low')
y = df['Low']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_regressor.fit(X_train, y_train)
y_pred = gb_regressor.predict(X_test)
y_pred = gb_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
# Evaluate the model
print(y_test, y_pred)



# Assuming you've already trained the model (e.g., gb_regressor)
feature_importances = gb_regressor.feature_importances_
feature_names = X.columns

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['Close*'])
plt.xlabel('Timestamp')
plt.ylabel('Close Prices')
plt.title('Close Prices Over Time')
plt.show()

r2 = r2_score(y_test, y_pred)
print(f'R² score: {r2:.2f}')
from sklearn.metrics import mean_absolute_error

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Calculate Mean Absolute Percentage Error (MAPE)
mape = 100 * (abs(y_test - y_pred) / y_test).mean()
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# STOCK PRICE TRENDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("C:\\Users\\Thanu\OneDrive\\Documents\\Desktop\\yahoofinance.csv")
# Display basic statistics
print(df.describe())

df = pd.read_csv("C:\\Users\\Thanu\\OneDrive\\Documents\\Desktop\\yahoofinance.csv")

# Convert the "ds" column to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Extract timestamps (in seconds) from the datetime column
df['ds'] = df['ds'].astype('int64') // 10**9


# Visualize stock price trends
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['Open'], label='Open')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(' Price Trends')
plt.legend()
plt.show()


# Visualize stock price trends
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['High'], label='High')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(' Price Trends')
plt.legend()
plt.show()


# Visualize stock price trends
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['Low'], label='Low')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(' Price Trends')
plt.legend()
plt.show()



# Visualize stock price trends
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['Close*'], label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(' Price Trends')
plt.legend()
plt.show()



# Visualize stock price trends
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['Adj Close**'], label=' Adjacent Closing Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(' Price Trends')
plt.legend()
plt.show()


# Visualize stock price trends
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['Volume'], label='Volume')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(' Price Trends')
plt.legend()
plt.show()

from tensorflow.keras.layers import Dropout
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
valid_data = pd.read_csv("C:\\Users\\Thanu\OneDrive\\Documents\\Desktop\\yahoofinance.csv")
valid_data['ds'] = pd.to_datetime(valid_data['ds'])
valid_data['ds'] = valid_data['ds'].astype('int64') / 10**9
print(valid_data.dtypes)
dates = valid_data['ds']
X = valid_data.drop(['Close*'], axis=1)
y = valid_data['Close*']
actual_prices = valid_data['Close*']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
actual_changes = np.diff(actual_prices)

# Create a new DataFrame with the dates and changes
changes_df = pd.DataFrame({'ds': dates[:-1], 'y': actual_changes})

# Train the model using GRU
regressorGRU = Sequential()
regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressorGRU.add(Dropout(0.2))
regressorGRU.add(GRU(units=50, return_sequences=True))
regressorGRU.add(Dropout(0.2))
regressorGRU.add(GRU(units=50, return_sequences=True))
regressorGRU.add(Dropout(0.2))
regressorGRU.add(GRU(units=50))
regressorGRU.add(Dropout(0.2))
regressorGRU.add(Dense(units=1))

regressorGRU.compile(optimizer='adam',loss='mean_squared_error')
regressorGRU.fit(X_train,y_train,epochs=100,batch_size=10)

# Make predictions using the test set
predictions_changes = regressorGRU.predict(X_test)

# Convert the changes back to actual values
predictions_df = pd.DataFrame({'ds': dates[len(predictions_changes):], 'y': actual_prices[-len(predictions_changes):] + np.cumsum(predictions_changes)})

# Plot the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(dates, actual_prices, label='Actual Prices', color='blue')
plt.plot(dates[:len(predictions_df)], predictions_df, label='Predicted Prices', color='orange')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.grid(True)
plt.show()

y_pred = regressorGRU.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# LINEAR REGRESSION AND XGBREGRESSOR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("C:\\Users\\Thanu\OneDrive\\Documents\\Desktop\\yahoofinance.csv")
# Display basic statistics
print(df.describe())

df = pd.read_csv("C:\\Users\\Thanu\\OneDrive\\Documents\\Desktop\\yahoofinance.csv")

# Convert the "ds" column to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Extract timestamps (in seconds) from the datetime column
df['ds'] = df['ds'].astype('int64') // 10**9
train, valid = train_test_split(df, test_size=0.2, shuffle=False)
# Initialize models
lr_model = LinearRegression()
xgb_model = XGBRegressor()


# Fit models
lr_model.fit(train[['ds']], train['Open'])
xgb_model.fit(train[['ds']], train['Open'])
# Make predictions
lr_pred = lr_model.predict(valid[['ds']])
xgb_pred = xgb_model.predict(valid[['ds']])

lr_mse = mean_squared_error(valid['Open'], lr_pred)
xgb_mse = mean_squared_error(valid['Open'], xgb_pred)
lr_r2 = r2_score(valid['Open'], lr_pred)
xgb_r2 = r2_score(valid['Open'], xgb_pred)

print(f"Linear Regression - MSE: {lr_mse:.2f}, R2: {lr_r2:.2f}")
print(f"XGBoost - MSE: {xgb_mse:.2f}, R2: {xgb_r2:.2f}")
plt.figure(figsize=(10, 6))
plt.plot(valid['ds'], valid['Open'], label='Actual Price')
plt.plot(valid['ds'], lr_pred, label='LR Prediction')
plt.plot(valid['ds'], xgb_pred, label='XGB Prediction')
plt.xlabel('Date')
plt.ylabel('Open')
plt.title('Stock Price Predictions')
plt.legend()
plt.show()



# Fit models
lr_model.fit(train[['ds']], train['High'])
xgb_model.fit(train[['ds']], train['High'])
# Make predictions
lr_pred = lr_model.predict(valid[['ds']])
xgb_pred = xgb_model.predict(valid[['ds']])

lr_mse = mean_squared_error(valid['High'], lr_pred)
xgb_mse = mean_squared_error(valid['High'], xgb_pred)
lr_r2 = r2_score(valid['High'], lr_pred)
xgb_r2 = r2_score(valid['High'], xgb_pred)


print(f"Linear Regression - MSE: {lr_mse:.2f}, R2: {lr_r2:.2f}")
print(f"XGBoost - MSE: {xgb_mse:.2f}, R2: {xgb_r2:.2f}")
plt.figure(figsize=(10, 6))
plt.plot(valid['ds'], valid['Low'], label='Actual Price')
plt.plot(valid['ds'], lr_pred, label='LR Prediction')
plt.plot(valid['ds'], xgb_pred, label='XGB Prediction')
plt.xlabel('Date')
plt.ylabel('High')
plt.title('Stock Price Predictions')
plt.legend()
plt.show()


# Fit models
lr_model.fit(train[['ds']], train['Low'])
xgb_model.fit(train[['ds']], train['Low'])
# Make predictions
lr_pred = lr_model.predict(valid[['ds']])
xgb_pred = xgb_model.predict(valid[['ds']])

lr_mse = mean_squared_error(valid['Low'], lr_pred)
xgb_mse = mean_squared_error(valid['Low'], xgb_pred)
lr_r2 = r2_score(valid['Low'], lr_pred)
xgb_r2 = r2_score(valid['Low'], xgb_pred)

print(f"Linear Regression - MSE: {lr_mse:.2f}, R2: {lr_r2:.2f}")
print(f"XGBoost - MSE: {xgb_mse:.2f}, R2: {xgb_r2:.2f}")
plt.figure(figsize=(10, 6))
plt.plot(valid['ds'], valid['Low'], label='Actual Price')
plt.plot(valid['ds'], lr_pred, label='LR Prediction')
plt.plot(valid['ds'], xgb_pred, label='XGB Prediction')
plt.xlabel('Date')
plt.ylabel('Low')
plt.title('Stock Price Predictions')
plt.legend()
plt.show()

# Fit models
lr_model.fit(train[['ds']], train['Adj Close**'])
xgb_model.fit(train[['ds']], train['Adj Close**'])
# Make predictions
lr_pred = lr_model.predict(valid[['ds']])
xgb_pred = xgb_model.predict(valid[['ds']])

# Evaluate
lr_mse = mean_squared_error(valid['Adj Close**'], lr_pred)
xgb_mse = mean_squared_error(valid['Adj Close**'], xgb_pred)
lr_r2 = r2_score(valid['Adj Close**'], lr_pred)
xgb_r2 = r2_score(valid['Adj Close**'], xgb_pred)

print(f"Linear Regression - MSE: {lr_mse:.2f}, R2: {lr_r2:.2f}")
print(f"XGBoost - MSE: {xgb_mse:.2f}, R2: {xgb_r2:.2f}")
plt.figure(figsize=(10, 6))
plt.plot(valid['ds'], valid['Adj Close**'], label='Actual Price')
plt.plot(valid['ds'], lr_pred, label='LR Prediction')
plt.plot(valid['ds'], xgb_pred, label='XGB Prediction')
plt.xlabel('Date')
plt.ylabel('Adjacent Close')
plt.title('Stock Price Predictions')
plt.legend()
plt.show()


# Fit models
lr_model.fit(train[['ds']], train['Close*'])
xgb_model.fit(train[['ds']], train['Close*'])
# Make predictions
lr_pred = lr_model.predict(valid[['ds']])
xgb_pred = xgb_model.predict(valid[['ds']])

# Evaluate
lr_mse = mean_squared_error(valid['Close*'], lr_pred)
xgb_mse = mean_squared_error(valid['Close*'], xgb_pred)
lr_r2 = r2_score(valid['Close*'], lr_pred)
xgb_r2 = r2_score(valid['Close*'], xgb_pred)

print(f"Linear Regression - MSE: {lr_mse:.2f}, R2: {lr_r2:.2f}")
print(f"XGBoost - MSE: {xgb_mse:.2f}, R2: {xgb_r2:.2f}")
plt.figure(figsize=(10, 6))
plt.plot(valid['ds'], valid['Close*'], label='Actual Price')
plt.plot(valid['ds'], lr_pred, label='LR Prediction')
plt.plot(valid['ds'], xgb_pred, label='XGB Prediction')
plt.xlabel('Date')
plt.ylabel('Close')
plt.title('Stock Price Predictions')
plt.legend()
plt.show()



# Fit models
lr_model.fit(train[['ds']], train['Volume'])
xgb_model.fit(train[['ds']], train['Volume'])
# Make predictions
lr_pred = lr_model.predict(valid[['ds']])
xgb_pred = xgb_model.predict(valid[['ds']])

lr_mse = mean_squared_error(valid['Volume'], lr_pred)
xgb_mse = mean_squared_error(valid['Volume'], xgb_pred)
lr_r2 = r2_score(valid['Volume'], lr_pred)
xgb_r2 = r2_score(valid['Volume'], xgb_pred)

print(f"Linear Regression - MSE: {lr_mse:.2f}, R2: {lr_r2:.2f}")
print(f"XGBoost - MSE: {xgb_mse:.2f}, R2: {xgb_r2:.2f}")
plt.figure(figsize=(10, 6))
plt.plot(valid['ds'], valid['Volume'], label='Actual Price')
plt.plot(valid['ds'], lr_pred, label='LR Prediction')
plt.plot(valid['ds'], xgb_pred, label='XGB Prediction')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Stock Price Predictions')
plt.legend()
plt.show()
