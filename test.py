import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df_2021 = pd.read_csv('ganga_2021.csv')
df_2021 = df_2021.apply(pd.to_numeric, errors='coerce')
df_2021.fillna(df_2021.mean(), inplace=True)
grouped_df_2021 = df_2021.groupby(['Station Code', 'Year'], as_index=False).mean()


features = ['DO (mg/L)', 'BOD (mg/L)', 'FC (MPN/100ml)', 'FS (MPN/100ml)']
X = df_2021[features]

print("Contaminants list\n1. DO (mg/L)\n2. BOD (mg/L)\n3. FC (MPN/100ml)\n4. FS (MPN/100ml)")
contaminate = int(input("Enter contaminant for plotting linear regression:"))

predictions_list = []

for station_code in grouped_df_2021['Station Code'].unique():
    station_data = grouped_df_2021[grouped_df_2021['Station Code'] == station_code]

    year_to_predict = station_data['Year'].max() + 1

    X_train, y_train = station_data[features], station_data[features[contaminate - 1]]
    model = LinearRegression()
    model.fit(X_train, y_train)

    features_next_year = df_2021[df_2021['Station Code'] == station_code].tail(1)[features]
    predicted_contamination_level = model.predict(features_next_year)

    predictions_list.append({
        'Station Code': station_code,
        'PredictedContaminationLevel': predicted_contamination_level[0]
    })


predictions_df = pd.DataFrame(predictions_list)


df_2022 = pd.read_csv('ganga_2022.csv')
df_2022 = df_2022.apply(pd.to_numeric, errors='coerce')
df_2022.fillna(df_2022.mean(), inplace=True)


merged_df = pd.merge(predictions_df, df_2022, how='inner', on='Station Code', suffixes=('_predicted', '_actual'))

print(merged_df)


plt.figure(figsize=(10, 6))
plt.scatter(merged_df['PredictedContaminationLevel'], merged_df[features[contaminate - 1]], c=merged_df[features[contaminate - 1]], cmap='Blues', label='Actual')
plt.scatter(merged_df['PredictedContaminationLevel'], merged_df['PredictedContaminationLevel'], c='red', label='Predicted')

plt.xlabel('Predicted Contamination Level')
plt.ylabel('Actual Contamination Level')
plt.title('Linear Regression Model Predictions vs Actual Values')
plt.legend()
plt.colorbar(label='Contamination Level')  # Add colorbar to show contamination level
plt.show()
