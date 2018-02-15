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

df = pd.read_csv(TRAIN_FILEPATH)

grouped = df.groupby(df.YrSold)
grouped.SalePrice = grouped.SalePrice.aggregate([np.mean, np.std])
mean_price_per_year = dict(zip(list(grouped.YrSold.groups.keys()), grouped.SalePrice["mean"]))

model_df = df[["SalePrice", "LotArea", "TotRmsAbvGrd", "GarageArea", "YrSold"]]
model_df.SalePrice = model_df.SalePrice.apply(np.log)
model_df.LotArea = model_df.LotArea.apply(np.log)
model_df.YrSold = model_df.YrSold.map(mean_price_per_year)

train_df, test_df = train_test_split(model_df)
x_train = train_df[["LotArea", "TotRmsAbvGrd", "GarageArea", "YrSold"]]
y_train = train_df.SalePrice
x_test = test_df[["LotArea", "TotRmsAbvGrd", "GarageArea", "YrSold"]]
y_test = test_df.SalePrice

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
err = mean_squared_error(y_test, y_pred)

print("Variance explained = {}".format(round(r2, 4)))
print("Root mean squared error = {}".format(round(err, 4)))
plt.figure()
plt.plot(y_pred, y_test, "o")
plt.show()
