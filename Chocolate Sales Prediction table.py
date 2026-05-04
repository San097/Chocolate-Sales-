import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("/content/drive/MyDrive/Chocolate Sales.csv")

print(df.head())
print(df.info())

df['Amount'] = df['Amount'].replace({r'\$': '', r',': ''}, regex=True).astype(float)

df = df.dropna()

original_categorical_data = {}
label_encoders = {}

for col in df.select_dtypes(include='object').columns:
    original_categorical_data[col] = df[col].unique().tolist()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_column = 'Amount'

X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

sample = X.iloc[0:1]
prediction = model.predict(sample)
print("Sample Prediction:", prediction)
