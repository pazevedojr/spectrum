import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PKG_DIR, "data")
TRAIN_FILEPATH = os.path.join(DATA_DIR, "train.csv")
TEST_FILEPATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILEPATH = os.path.join(DATA_DIR, "submission.csv")

train_df = pd.read_csv(TRAIN_FILEPATH)

grouped = train_df.groupby(train_df.YrSold)
grouped.SalePrice = grouped.SalePrice.aggregate([np.mean, np.std])
mean_price_per_year = dict(zip(list(grouped.YrSold.groups.keys()), grouped.SalePrice["mean"]))

train_df = train_df[["SalePrice", "LotArea", "YrSold", "OverallCond", "OverallQual", "GarageCars", "GrLivArea", "FullBath"]]
train_df.SalePrice = train_df.SalePrice.apply(np.log)
train_df.LotArea = train_df.LotArea.apply(np.log)
train_df.YrSold = train_df.YrSold.map(mean_price_per_year)

x_train = train_df[["LotArea", "YrSold", "OverallCond", "OverallQual", "GarageCars", "GrLivArea", "FullBath"]]
y_train = train_df.SalePrice

model = LinearRegression()
model.fit(x_train, y_train)

test_df = pd.read_csv(TEST_FILEPATH)
test_df = test_df[["Id", "LotArea", "YrSold", "OverallCond", "OverallQual", "GarageCars", "GrLivArea", "FullBath"]]
test_df = test_df.fillna(0)
test_df.LotArea = test_df.LotArea.apply(np.log)
test_df.YrSold = test_df.YrSold.map(mean_price_per_year)
pred = model.predict(test_df[["LotArea", "YrSold", "OverallCond", "OverallQual", "GarageCars", "GrLivArea", "FullBath"]])

pred = [p for p in np.exp(pred)]
ids = [id_ for id_ in test_df.Id]
data = [(id_, pred) for id_, pred in zip(ids, pred)]

submission_df = pd.DataFrame(data=data, columns=["Id", "SalePrice"])
submission_df.to_csv(SUBMISSION_FILEPATH, index=False)
