#step 1 . read the data 
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)



# you can save the boxplot
plt. figure(figsize=(6, 4)) 
df['MedHouseVal'].plot.box()
plt.title("Boxplot of Median House Value")
plt.ylabel("Median House Value")
plt. tight_layout ()
plt.savefig("figures/med_house_value_boxplot.png")
# <-- saved file
plt.close()



import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt


# 1. Load California Housing dataset
data = fetch_california_housing(as_frame=True)

X = data.frame.drop(columns=["MedHouseVal"])
y = data.frame["MedHouseVal"]

print(X.head())
print(y.head())
