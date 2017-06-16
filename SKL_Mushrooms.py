#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

train_df = pd.read_csv("input/mushrooms.csv")
#%%
le = LabelEncoder()
for col in train_df.columns:
    train_df[col] = le.fit_transform(train_df[col])

X = train_df.iloc[:, 1:23]
Y = train_df.iloc[:, :1]

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
#%%
model = joblib.load("ShroomsMLP.pkl")

predic = model.predict(train_x)
print("Model Prediction: ", predic)
print("Real Value: ", train_y)
