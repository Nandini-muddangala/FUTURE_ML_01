import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("data/Sample - Superstore.csv", encoding='latin1')

# Convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])
df = df.sort_values('Order Date')

# ----------------------------
# 2. Create Monthly Sales
# ----------------------------
monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()

# ----------------------------
# 3. Feature Engineering
# ----------------------------
monthly_sales['Month'] = monthly_sales['Order Date'].dt.month
monthly_sales['Year'] = monthly_sales['Order Date'].dt.year
monthly_sales['Lag_1'] = monthly_sales['Sales'].shift(1)

monthly_sales = monthly_sales.dropna()

# ----------------------------
# 4. Features and Target
# ----------------------------
X = monthly_sales[['Month', 'Year', 'Lag_1']]
y = monthly_sales['Sales']

# ----------------------------
# 5. Train-Test Split (Time-Based)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ----------------------------
# 6. Train Model
# ----------------------------
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 7. Test Predictions
# ----------------------------
y_pred = model.predict(X_test)

# ----------------------------
# 8. Model Evaluation
# ----------------------------
print("Model Evaluation Results:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ----------------------------
# 9. Future Forecast (Next 6 Months)
# ----------------------------
last_date = monthly_sales['Order Date'].max()
future_dates = pd.date_range(start=last_date, periods=7, freq='M')[1:]

future_df = pd.DataFrame({'Order Date': future_dates})
future_df['Month'] = future_df['Order Date'].dt.month
future_df['Year'] = future_df['Order Date'].dt.year

last_sales = monthly_sales['Sales'].iloc[-1]
future_predictions = []

for i in range(len(future_df)):
    input_features = [[
        future_df.iloc[i]['Month'],
        future_df.iloc[i]['Year'],
        last_sales
    ]]
    
    pred = model.predict(input_features)[0]
    future_predictions.append(pred)
    
    last_sales = pred

future_df['Predicted Sales'] = future_predictions

# ----------------------------
# 10. Final Graph with Labels
# ----------------------------
plt.figure(figsize=(12,6))

# Historical Sales (Blue)
plt.plot(
    monthly_sales['Order Date'],
    monthly_sales['Sales'],
    color='blue',
    label='Historical Sales'
)

# Test Predictions (Green)
plt.plot(
    monthly_sales['Order Date'].iloc[-len(y_test):],
    y_pred,
    color='green',
    label='Test Predictions'
)

# Future Forecast (Red Dashed)
plt.plot(
    future_df['Order Date'],
    future_df['Predicted Sales'],
    color='red',
    linestyle='dashed',
    linewidth=2,
    label='Future Forecast'
)

# Text labels directly on graph
plt.text(
    monthly_sales['Order Date'].iloc[5],
    monthly_sales['Sales'].iloc[5],
    "Historical Sales",
    color='blue'
)

plt.text(
    monthly_sales['Order Date'].iloc[-1],
    y_pred[-1],
    "Test Prediction",
    color='green'
)

plt.text(
    future_df['Order Date'].iloc[-1],
    future_df['Predicted Sales'].iloc[-1],
    "Future Forecast",
    color='red'
)

plt.title("Sales Demand Forecasting - Complete Overview")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
