import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rossmann Store Sales Prediction",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "rossmann_sales_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "rossmann_sales_model.pkl was not found. "
        "Please place the model file in the same folder as app.py."
    )
    st.stop()

model = joblib.load(MODEL_PATH)


# ============================================================
# TITLE
# ============================================================

st.title("📊 Rossmann Store Sales Prediction")

st.write(
    "Enter the store and promotion details to predict sales."
)


# ============================================================
# STORE INFORMATION
# ============================================================

st.header("Store Information")

store = st.number_input(
    "Store ID",
    min_value=1,
    max_value=1115,
    value=1,
    step=1
)

day_of_week = st.selectbox(
    "Day of Week",
    [1, 2, 3, 4, 5, 6, 7]
)

customers = st.number_input(
    "Customers",
    min_value=0,
    value=500,
    step=1
)

open_store = st.selectbox(
    "Store Open",
    [0, 1],
    index=1
)

promo = st.selectbox(
    "Promotion",
    [0, 1]
)

school_holiday = st.selectbox(
    "School Holiday",
    [0, 1]
)


# ============================================================
# HOLIDAY
# ============================================================

state_holiday = st.selectbox(
    "State Holiday",
    ["0", "a", "b", "c"]
)


# ============================================================
# STORE TYPE
# ============================================================

store_type = st.selectbox(
    "Store Type",
    ["a", "b", "c", "d"]
)

assortment = st.selectbox(
    "Assortment",
    ["a", "b", "c"]
)


# ============================================================
# COMPETITION INFORMATION
# ============================================================

st.header("Competition Information")

competition_distance = st.number_input(
    "Competition Distance",
    min_value=0.0,
    value=500.0,
    step=10.0
)

competition_open_month = st.number_input(
    "Competition Open Since Month",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)

competition_open_year = st.number_input(
    "Competition Open Since Year",
    min_value=1900,
    max_value=2030,
    value=2010,
    step=1
)


# ============================================================
# PROMO 2 INFORMATION
# ============================================================

st.header("Promo2 Information")

promo2 = st.selectbox(
    "Promo2",
    [0, 1]
)

promo2_since_week = st.number_input(
    "Promo2 Since Week",
    min_value=1,
    max_value=52,
    value=1,
    step=1
)

promo2_since_year = st.number_input(
    "Promo2 Since Year",
    min_value=2000,
    max_value=2030,
    value=2010,
    step=1
)

promo_interval = st.selectbox(
    "Promo Interval",
    ["None", "Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec"]
)

if promo_interval == "None":
    promo_interval = None


# ============================================================
# DATE INFORMATION
# ============================================================

st.header("Date Information")

year = st.number_input(
    "Year",
    min_value=2013,
    max_value=2030,
    value=2015,
    step=1
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=1,
    step=1
)

week_of_year = st.number_input(
    "Week of Year",
    min_value=1,
    max_value=53,
    value=1,
    step=1
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("🔮 Predict Sales"):

    input_data = pd.DataFrame({
        "Store": [store],
        "DayOfWeek": [day_of_week],
        "Customers": [customers],
        "Open": [open_store],
        "Promo": [promo],
        "StateHoliday": [state_holiday],
        "SchoolHoliday": [school_holiday],
        "StoreType": [store_type],
        "Assortment": [assortment],
        "CompetitionDistance": [competition_distance],
        "CompetitionOpenSinceMonth": [competition_open_month],
        "CompetitionOpenSinceYear": [competition_open_year],
        "Promo2": [promo2],
        "Promo2SinceWeek": [promo2_since_week],
        "Promo2SinceYear": [promo2_since_year],
        "PromoInterval": [promo_interval],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "WeekOfYear": [week_of_year]
    })

    try:

        prediction = model.predict(input_data)

        predicted_sales = max(0, float(prediction[0]))

        st.success(
            f"### Predicted Sales: ₹{predicted_sales:,.2f}"
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Rossmann Store Sales Prediction | Random Forest Regression"
)