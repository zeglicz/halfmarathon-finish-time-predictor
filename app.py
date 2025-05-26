import pandas as pd
import streamlit as st
from typing import Optional
from pydantic import BaseModel, Field
from pycaret.regression import load_model, predict_model

import instructor
from dotenv import dotenv_values
from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import OpenAI

st.set_page_config(page_title="halfmarathon-finish-time-predictor", layout="centered")
st.title("halfmarathon-finish-time-predictor")

tab1, tab2 = st.tabs(["Manual Predictor", "Smart Predictor"])

#
# PREDICTOR
#

model = load_model("./models/halfmarathon_predictor")

env = dotenv_values(".env")

langfuse = Langfuse(
    public_key=env.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=env.get("LANGFUSE_SECRET_KEY"),
    host=env.get("LANGFUSE_HOST"),
)


class InputData(BaseModel):
    time_5k_sec: Optional[float] = Field(..., description="Time to run 5km in seconds")
    pace_5k_sec: Optional[float] = Field(
        ..., description="Pace per kilometer in seconds"
    )
    gender: Optional[int] = Field(
        ..., description="Gender as integer: 0=female, 1=male"
    )
    age: Optional[float] = Field(..., description="Age in years")


def get_openai_client():
    client = OpenAI(api_key=st.session_state["openai_api_key"])
    return instructor.patch(client)


@observe()
def extract_input_from_text(user_text: str) -> InputData | None:
    try:
        openai_client = get_openai_client()

        # langfuse.trace(name="test-trace", input={"user_text": user_text})
        parsed_data = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Extract running information from: '{user_text}'",
                }
            ],
            response_model=InputData,
            max_retries=2,
        )

        return parsed_data
    except Exception as e:
        st.error(f":warning: Error extracting data. Try rephrasing your input.")
        st.exception(e)
        return None


def time_to_seconds(t):
    try:
        parts = list(map(int, t.strip().split(":")))
        if len(parts) == 2:
            minutes, seconds = parts
            if seconds >= 60:
                return None
            hours = 0
        else:
            return None
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None


#
# MAIN
#

with tab1:
    st.header(":bar_chart: Enter Your Running Data")

    time_5k_str = st.text_input("Time for 5 kilometer run (MM:SS)", "27:43")
    age = st.number_input("Age", min_value=6, max_value=120, value=40)
    gender_str = st.selectbox("Gender", ["Female", "Male"])

    time_5k = time_to_seconds(time_5k_str)
    gender = 0 if gender_str == "Female" else 1

    if st.button("Predict Half Marathon Time"):
        if time_5k is not None:
            pace_5k = time_5k / 5
            input_data = pd.DataFrame(
                [
                    {
                        "time_5k_sec": time_5k,
                        "pace_5k_sec": pace_5k,
                        "gender": gender,
                        "age": float(age),
                    }
                ]
            )

            prediction = predict_model(model, data=input_data)
            predicted_seconds = prediction["prediction_label"].iloc[0]

            hours = int(predicted_seconds // 3600)
            minutes = int((predicted_seconds % 3600) // 60)
            seconds = int(predicted_seconds % 60)

            st.success(
                f":checkered_flag: Estimated Half Marathon Time: {hours}h {minutes}m {seconds}s"
            )
        else:
            st.error(":warning: Invalid time format. Use MM:SS")

with tab2:
    st.header(":robot_face: AI – Let the model understand you")

    if not st.session_state.get("openai_api_key"):
        if "OPENAI_API_KEY" in env:
            st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
        else:
            st.info("Add your OpenAI API key to use the application")
            st.session_state["openai_api_key"] = st.text_input("API key")

    if not st.session_state.get("openai_api_key"):
        st.stop()
        st.error("Invalid OpenAI API key")

    openai_client = OpenAI(api_key=st.session_state["openai_api_key"])

    st.markdown(
        "Describe your **age**, **gender**, and **5 kilometer run time** (e.g., *I'm 40 years old, male, and run 5 kilometers in 27:43 minutes*)."
    )

    user_input = st.text_area("Enter your description:")

    if st.button("Submit to AI"):
        if user_input.strip():
            with st.spinner("Thinking..."):
                result = extract_input_from_text(user_input)
                if result:

                    # Abort prediction if the LLM output contains empty or missing fields
                    result_dict = result.model_dump()
                    invalid_keys = [k for k, v in result_dict.items() if not v]
                    if invalid_keys:
                        st.error(
                            f":warning: Empty or invalid fields: {', '.join(invalid_keys)}"
                        )
                        raise ValueError(f"Validation failed for: {invalid_keys}")

                    st.success(":white_check_mark: Data extracted successfully:")
                    st.json(result.model_dump())

                    input_df = pd.DataFrame([result.model_dump()])
                    prediction = predict_model(model, data=input_df)
                    predicted = prediction["prediction_label"].iloc[0]

                    h = int(predicted // 3600)
                    m = int((predicted % 3600) // 60)
                    s = int(predicted % 60)

                    st.success(
                        f":checkered_flag: Estimated Half Marathon Time: {h}h {m}m {s}s"
                    )
