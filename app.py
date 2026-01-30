import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# App Title
st.title("Smart Sales Profit Prediction App")
st.markdown("""
Predict future sales and profits using your sales data. Upload a CSV file with columns:
`Date`, `Units_Sold`, `Price`, `Cost`.
""")

# Sidebar: Upload CSV
st.sidebar.header("Upload Sales Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Read CSV
    data = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.subheader("Sales Data")
    st.dataframe(data)

    # Prepare features and target
    X = data.index.values.reshape(-1, 1)  # Using row index as a simple time feature
    y = data["Units_Sold"]

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 7 days
    future_index = pd.DataFrame(range(len(data), len(data) + 7))
    predicted_sales = model.predict(future_index)

    # Estimate profit
    avg_price = data["Price"].mean()
    avg_cost = data["Cost"].mean()
    predicted_profit = (avg_price - avg_cost) * predicted_sales

    # Show predictions
    st.subheader("Next 7 Days Sales Prediction")
    prediction_df = pd.DataFrame({
        "Day": range(1, 8),
        "Predicted_Units_Sold": predicted_sales.astype(int),
        "Estimated_Profit": predicted_profit.astype(int)
    })
    st.dataframe(prediction_df)

    # Plot sales prediction
    plt.figure(figsize=(10,5))
    plt.plot(range(len(data)), y, label="Actual Sales")
    plt.plot(range(len(data), len(data)+7), predicted_sales, label="Predicted Sales", linestyle="--", color="orange")
    plt.xlabel("Days")
    plt.ylabel("Units Sold")
    plt.title("Sales Prediction")
    plt.legend()
    st.pyplot(plt)

else:
    st.info("Please upload a CSV file to see predictions.")
