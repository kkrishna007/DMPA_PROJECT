import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv('Ganga_Water_Quality.csv')

df = df.apply(pd.to_numeric, errors='coerce')

df.fillna(df.mean(), inplace=True)

features = ['DO (mg/L)', 'BOD (mg/L)', 'FC (MPN/100ml)', 'FS (MPN/100ml)']
X = df[features]
print("Contaminates list\n1. DO (mg/L)\n2. BOD (mg/L)\n3. FC (MPN/100ml)\n4. FS (MPN/100ml)")
contaminate=int(input("Enter contaminate for plotting linear regression:"))
y = df[features[contaminate-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=33)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Water Contamination Level')
plt.ylabel('Predicted Water Contamination Level')
plt.title('Actual vs Predicted Water Contamination Level')
plt.show()
