# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
from io import BytesIO

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(page_title="Smart Sales Profit Prediction App", layout="wide")
st.title("💰 Smart Sales Profit Prediction App")

# -------------------------------
# Currency Selection & Conversion
# -------------------------------
st.sidebar.header("Currency Conversion")
currency_options = ["USD", "EUR", "KES", "GBP"]
currency = st.sidebar.selectbox("Select Currency for Profit Display:", currency_options)

# Static demo rates
currency_rates = {"USD": 1, "EUR": 0.92, "KES": 150, "GBP": 0.81}
rate = currency_rates[currency]

# -------------------------------
# Dataset Upload / Built-in Dataset
# -------------------------------
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()
else:
    st.info("Using built-in sample dataset")
    np.random.seed(42)
    data = pd.DataFrame({
        "Units_Sold": np.random.randint(50, 500, 100),
        "Marketing_Spend": np.random.randint(1000, 10000, 100),
        "Price_per_Unit": np.random.randint(20, 100, 100),
    })
    data["Profit"] = data["Units_Sold"] * data["Price_per_Unit"] * 0.2 + data["Marketing_Spend"] * 0.1

st.subheader("Sample Data")
st.dataframe(data.head())

# -------------------------------
# Feature Selection & Model Training
# -------------------------------
st.header("📊 Linear Regression Model Training")
features = st.multiselect(
    "Select Features for Prediction:", options=list(data.columns.drop("Profit")), 
    default=list(data.columns.drop("Profit"))
)
target = "Profit"

if len(features) == 0:
    st.warning("Select at least one feature to train the model.")
    st.stop()

X = data[features]
y = data[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Metrics")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

    # Feature importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    st.dataframe(importance)
    fig_importance = px.bar(importance, x="Feature", y="Coefficient", title="Feature Importance")
    st.plotly_chart(fig_importance)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

# -------------------------------
# Predict Profit Section
# -------------------------------
st.header("💹 Predict Profit")
input_data = {}
for feature in features:
    min_val = int(data[feature].min())
    max_val = int(data[feature].max())
    input_data[feature] = st.number_input(f"Enter {feature}:", min_value=min_val, max_value=max_val, value=min_val)

if st.button("Predict Profit"):
    try:
        input_df = pd.DataFrame([input_data])
        pred_profit = model.predict(input_df)[0] * rate
        st.success(f"Predicted Profit in {currency}: {pred_profit:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# Trend Plot for Units Sold
# -------------------------------
st.header("📈 Units Sold Trend")
try:
    fig_trend = px.line(data, y="Units_Sold", title="Units Sold Trend", labels={"index": "Index", "Units_Sold": "Units Sold"})
    st.plotly_chart(fig_trend)
except Exception as e:
    st.error(f"Failed to generate trend plot: {e}")

# -------------------------------
# Downloadable Prediction Report
# -------------------------------
st.header("📄 Download Prediction Report")
if st.button("Generate Report"):
    try:
        report = data.copy()
        report["Profit_in_Selected_Currency"] = report["Profit"] * rate
        buffer = BytesIO()
        report.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("Download CSV Report", buffer, file_name="prediction_report.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Report generation failed: {e}")

# -------------------------------
# YouTube Advice / Info Links
# -------------------------------
st.header("🎓 YouTube Advice & Tips")
youtube_links = {
    "Linear Regression Basics": "https://www.youtube.com/watch?v=J_LnPL3Qg70",
    "Feature Importance Explained": "https://www.youtube.com/watch?v=2BXuAGLIa0g",
    "Profit Prediction Tutorials": "https://www.youtube.com/watch?v=ZkjP5RJLQF4",
}
for name, link in youtube_links.items():
    st.markdown(f'<a href="{link}" target="_blank">{name}</a>', unsafe_allow_html=True)
