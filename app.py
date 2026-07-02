import streamlit as st
import joblib

# Load model and encoder
model = joblib.load("company_model.pkl")
encoder = joblib.load("encoder.pkl")

# Page settings
st.set_page_config(
    page_title="Company Success Predictor",
    page_icon="🏢",
    layout="centered"
)

st.title("🏢 Company Success Prediction")
st.write("Enter the company details below.")

# Inputs
company = st.text_input("Company Name")

employees = st.number_input(
    "Number of Employees",
    min_value=1,
    step=1
)

investment = st.number_input(
    "Investment Amount",
    min_value=0.0,
    step=1000.0
)

# Predict button
if st.button("Predict"):

    try:
        company_encoded = encoder.transform([company])[0]

        result = model.predict([[company_encoded, employees, investment]])

        success = result[0][0]
        failure = result[0][1]
        survival = result[0][2]

        st.success("Prediction Completed!")

        st.subheader("Prediction Result")

        st.metric("Success Rate", f"{success:.2f}%")
        st.metric("Failure Rate", f"{failure:.2f}%")
        st.metric("Survival Rate", f"{survival:.2f}%")

        st.subheader("Business Status")

        if success >= 75:
            st.success("Highly Stable Company ✅")
            st.subheader("Recommendations")
            st.write("✅ Expand into new markets.")
            st.write("✅ Continue investing in innovation.")
            st.write("✅ Maintain customer satisfaction.")

        elif success >= 50:
            st.warning("Growing Company 🟡")
            st.subheader("Recommendations")
            st.write("⚠ Increase investment.")
            st.write("⚠ Improve marketing strategies.")
            st.write("⚠ Improve employee productivity.")

        else:
            st.error("Business Needs Improvement 🔴")
            st.subheader("Recommendations")
            st.write("❌ Increase investment.")
            st.write("❌ Recruit skilled employees.")
            st.write("❌ Improve financial planning.")

    except ValueError:
        st.error("Company name not found in the dataset.")