# Sales & Demand Forecasting - ML Task 1 (2026)

## Overview
In this project, I built a machine learning model to forecast monthly sales using historical business data.  
The idea is simple: businesses need to know what to expect in the coming months so they can plan inventory, staffing, and finances better.  

---

## Tools & Libraries
I used Python and some popular data science libraries:  
- Pandas & NumPy for data handling  
- Matplotlib for visualization  
- Scikit-learn for building the Random Forest model  

---

## Dataset
I worked with the **Superstore Sales Dataset** from [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final).  
It contains real sales data like Order Date, Sales amount, Product info, and Regions.  

---

## What I Did
- Aggregated sales by month  
- Created features like Month, Year, and previous month sales (Lag_1)  
- Split the data in a time-based way (no shuffling)  
- Trained a Random Forest model to predict future sales  
- Evaluated the model using MAE, RMSE, and R² score  
- Forecasted sales for the next 6 months  
- Visualized everything in clear, business-friendly graphs  

---

## Graphs Explained
- **Blue Line:** Actual historical sales  
- **Green Line:** Model predictions on recent data (to see how accurate it is)  
- **Red Dashed Line:** Future forecast for the next 6 months  

*(Insert a screenshot of your final graph here)*

---

## Why This Matters
With this forecast, a business can:  
- Plan inventory and avoid running out of stock  
- Adjust staffing based on expected sales  
- Plan marketing campaigns more effectively  
- Improve cash flow and financial planning  

Instead of guessing, the business can make data-driven decisions.

---

## How to Run
1. Open the `Sales_Forecasting.ipynb` notebook or the `sales_forecasting.py` script  
2. Install required packages:  
```bash
pip install -r requirements.txt