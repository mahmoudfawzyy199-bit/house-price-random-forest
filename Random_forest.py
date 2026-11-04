import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("C:/Users/Asus/Desktop/house_price_regression_dataset.csv")

############################### processing ################

# removing nulls
#print(df.isnull().sum())
df.dropna(inplace=True)
# removing dublicates
#print(df.duplicated().sum())
df = df.drop_duplicates()
#print(df.info())
s="Lot_Size"
Q1 = df[s].quantile(0.25)
Q3 = df[s].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df[s] < Q1 - 1.5*IQR) | (df[s] > Q3 + 1.5*IQR)]

# changing colomn values from ares to meter square
df["Lot_Size"] = df["Lot_Size"] * 4046.86

############################### Feature and target selection  ################
x = df.drop("House_Price", axis=1) # feature (all data set except y)
y = df["House_Price"]  # Target
############################### Train and test split ################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

############################### model training ################


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(x_train, y_train)

############################### output prediction  ################


y_pred = rf_model.predict(x_test) # the values that predicted of the x test (20%)

new_house = [[
    2000,   # Square_Footage
    3,      # Num_Bedrooms
    2,      # Num_Bathrooms
    2010,   # Year_Built
    8000,   # Lot_Size (m²)
    1,      # Garage_Size
    7       # Neighborhood_Quality
]]

new_prediction = rf_model.predict(new_house)
print("Predicted price for new house:", new_prediction[0])


############################### Evaluation (R square , MAE)  ################


r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {round(r2,3)} %")

mae = mean_absolute_error(y_test, y_pred) # mean of difference between actual values and predicted values
print("MAE:", mae)

############################### Visualization (Actual vs Predicted)  ################


plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Random Forest Regression: Actual vs Predicted")
plt.show()

