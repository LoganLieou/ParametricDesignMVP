from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# read in the training data then split into labels and features
training = pd.read_excel("./data/training.xlsx", sheet_name="Full information sample")
training = training.dropna(axis="columns") # there appeared to exist NA values
params = training.drop(["Observed GWP (assessed)"], axis=1)
labels = training["Observed GWP (assessed)"]

# output the number of entries the training set has
print("\ntraining dataset has: {0} entries".format(len(training)))

# Just fit the regression model
reg = RandomForestRegressor()
reg.fit(params, labels)

# Testing
testing = pd.read_excel("./data/testing.xlsx", sheet_name="Full information sample")
testing = testing.dropna(axis="columns")
testing_params = training.drop(["Observed GWP (assessed)"], axis=1)
testing_labels = training["Observed GWP (assessed)"]

# output information on parameters as well as test set accuracy
print("\nLIST OF PARAMETERS: \n")
for param in testing_params.columns: print(param)
print("\nTEST ACCURACY: ", reg.score(testing_params, testing_labels), "\n")

# Validation Set (this is sort of sus)
validation = pd.read_excel("./data/validation.xlsx", sheet_name="Full information sample")
validation = validation.dropna(axis="columns")
v_params = validation.drop(["Observed GWP (assessed)"], axis=1)
v_labels = validation["Observed GWP (assessed)"]
