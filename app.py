import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

model = load_model("./models/halfmarathon_predictor")

#
# PREDICTOR
#

# Convert inputs
def time_to_seconds(t):
    parts = list(map(int, t.strip().split(":")))
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        return None
    return hours * 3600 + minutes * 60 + seconds

#
# MAIN
#

st.title('halfmarathon-finish-time-predictor')

tab1, tab2 = st.tabs(['Manual Predictor','Smart Predictor'])

with tab1:
    st.header("Enter Your Running Data")

    time_5k_str = st.text_input("Time for 5 km (mm:ss)", "27:43")
    age = st.number_input("Age", min_value=10, max_value=100, value=40)
    gender_str = st.selectbox("Gender", ["Female", "Male"])

    time_5k = time_to_seconds(time_5k_str)
    pace_5k = time_5k / 5
    gender = 0 if gender_str == "Female" else 1

    if st.button("Predict Half Marathon Time"):
        if time_5k is not None:
            input_data = pd.DataFrame([{
                "time_5k_sec": time_5k,
                "pace_5k_sec": pace_5k,
                "gender": gender,
                "age": float(age)
            }])

            prediction = predict_model(model, data=input_data)
            predicted_seconds = prediction["prediction_label"].iloc[0]

            hours = int(predicted_seconds // 3600)
            minutes = int((predicted_seconds % 3600) // 60)
            seconds = int(predicted_seconds % 60)

            st.success(f"Estimated Half Marathon Time: {hours}h {minutes}m {seconds}s")
        else:
            st.error("Invalid time format. Use mm:ss")

with tab2:
    st.header("AI")