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
df.shape   # Get the dimensions of the dataset
df.describe()  # Summary statistics
df.info()