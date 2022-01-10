import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

training = pd.read_excel("./data/training.xlsx")
params = training.drop(["Observed GWP (assessed)"], axis=1)
labels = training["Observed GWP (assessed)"]

X_train, X_test, y_train, y_test = train_test_split(params, labels, test_size=0.2)

# Hyper Parameter Optimization
params = {"objective": "reg:squarederror",
          "max_depth": [0:10],
          "colsample_bylevel": 0.5,
          "learning_rate": [0.01:0.1:0.01],
          "random_state": [1:20]}

# Data
data = xgb.DMatrix(data=X_train, label=y_train)
cv_results = xgb.cv(dtrain)

# Using XGBoost for regression
reg = xgb.XGBRegressor()
reg.fit(X_train, y_train)
print("XGBoost: ", reg.score(X_test, y_test))

