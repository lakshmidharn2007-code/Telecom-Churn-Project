import streamlit as st
import pandas as pd
import joblib
import io
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Leaving Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

model = joblib.load("models/best_customer_leaving_model.joblib")

if "sample_data" not in st.session_state:
    st.session_state.sample_data = None

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

st.markdown("""
<style>
.main {
    background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}
.header-wrap {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 0.5rem;
}
.logo-box {
    width: 58px;
    height: 58px;
    border-radius: 16px;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    font-weight: 800;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.custom-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.2rem;
}
.custom-subtitle {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
    padding: 1.2rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}
.metric-label {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 0.35rem;
}
.metric-value {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
}
.low-risk {
    background: linear-gradient(135deg, #064e3b, #065f46);
}
.medium-risk {
    background: linear-gradient(135deg, #78350f, #92400e);
}
.high-risk {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
}
.reason-box, .strategy-box {
    background: rgba(255,255,255,0.05);
    border-left: 5px solid #38bdf8;
    padding: 1rem;
    border-radius: 14px;
    margin-bottom: 0.75rem;
    color: white;
}
.strategy-box {
    border-left: 5px solid #34d399;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.8rem 1rem;
    font-weight: 700;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
    color: white;
}
div[data-testid="stSidebar"] {
    background: #0f172a;
}
.sample-title {
    color: #cbd5e1;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

def get_risk_level(prob):
    if prob < 0.35:
        return "Low"
    elif prob < 0.70:
        return "Medium"
    return "High"

def get_confidence_score(prob):
    return max(prob, 1 - prob) * 100

def get_top_reasons(data):
    reasons = []

    if data["Contract"] == "Month-to-month":
        reasons.append("Short-term monthly plan increases leaving risk.")
    if data["Tenure Months"] < 12:
        reasons.append("The customer is still new, which increases leaving risk.")
    if data["Monthly Charges"] > 80:
        reasons.append("Higher monthly charges may reduce customer satisfaction.")
    if data["Internet Service"] == "Fiber optic":
        reasons.append("This internet plan is often linked with higher leaving risk.")
    if data["Tech Support"] == "No":
        reasons.append("No tech support may increase dissatisfaction when problems happen.")
    if data["Online Security"] == "No":
        reasons.append("No online security may show lower service engagement.")
    if data["Payment Method"] == "Electronic check":
        reasons.append("Electronic check payment is often linked with higher leaving behavior.")

    if not reasons:
        reasons.append("This customer profile shows stable stay-related characteristics.")

    return reasons[:3]

def get_retention_strategy(data):
    strategies = []

    if data["Contract"] == "Month-to-month":
        strategies.append("Offer a discount for switching to a longer plan.")
    if data["Monthly Charges"] > 80:
        strategies.append("Offer a lower-cost plan or a bundled value package.")
    if data["Tech Support"] == "No":
        strategies.append("Provide limited-time free tech support.")
    if data["Internet Service"] == "Fiber optic":
        strategies.append("Run a service-quality check and offer a loyalty benefit.")
    if data["Tenure Months"] < 12:
        strategies.append("Start an early retention campaign with welcome benefits.")
    if data["Payment Method"] == "Electronic check":
        strategies.append("Promote auto-pay with cashback or billing discounts.")

    if not strategies:
        strategies.append("Maintain engagement through loyalty rewards and regular follow-up.")

    return strategies[:3]

def load_sample(sample_name):
    samples = {
        "Low Risk": {
            "Gender": "Female",
            "Senior Citizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure Months": 60,
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "DSL",
            "Online Security": "Yes",
            "Online Backup": "Yes",
            "Device Protection": "Yes",
            "Tech Support": "Yes",
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Two year",
            "Paperless Billing": "No",
            "Payment Method": "Credit card (automatic)",
            "Monthly Charges": 64.50,
            "Total Charges": 3870.00
        },
        "Medium Risk": {
            "Gender": "Female",
            "Senior Citizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "Tenure Months": 18,
            "Phone Service": "Yes",
            "Multiple Lines": "Yes",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "Yes",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "No",
            "Streaming Movies": "Yes",
            "Contract": "One year",
            "Paperless Billing": "Yes",
            "Payment Method": "Bank transfer (automatic)",
            "Monthly Charges": 79.90,
            "Total Charges": 1438.20
        },
        "High Risk": {
            "Gender": "Male",
            "Senior Citizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "Tenure Months": 2,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "Fiber optic",
            "Online Security": "No",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "Yes",
            "Streaming Movies": "Yes",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
            "Monthly Charges": 99.50,
            "Total Charges": 199.00
        }
    }
    st.session_state.sample_data = samples[sample_name]

sample_defaults = st.session_state.sample_data if st.session_state.sample_data else {
    "Gender": "Male",
    "Senior Citizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "Yes",
    "Internet Service": "DSL",
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "Yes",
    "Tech Support": "Yes",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "One year",
    "Paperless Billing": "No",
    "Payment Method": "Credit card (automatic)",
    "Monthly Charges": 70.0,
    "Total Charges": 1000.0
}

st.markdown("""
<div class="header-wrap">
    <div class="logo-box">CL</div>
    <div>
        <div class="custom-title">Customer Leaving Prediction</div>
        <div class="custom-subtitle">Predict whether a customer may stay or leave, understand the reasons, and suggest the best retention action.</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Customer Inputs")
    st.markdown('<div class="sample-title">Load Sample Customers</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Low"):
            load_sample("Low Risk")
            st.rerun()
    with c2:
        if st.button("Mid"):
            load_sample("Medium Risk")
            st.rerun()
    with c3:
        if st.button("High"):
            load_sample("High Risk")
            st.rerun()

    gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(sample_defaults["Gender"]))
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], index=[0, 1].index(sample_defaults["Senior Citizen"]))
    partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(sample_defaults["Partner"]))
    dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(sample_defaults["Dependents"]))
    tenure_months = st.number_input("Tenure Months", min_value=0, max_value=100, value=int(sample_defaults["Tenure Months"]))
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(sample_defaults["Phone Service"]))
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=["Yes", "No", "No phone service"].index(sample_defaults["Multiple Lines"]))
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(sample_defaults["Internet Service"]))
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(sample_defaults["Online Security"]))
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(sample_defaults["Online Backup"]))
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(sample_defaults["Device Protection"]))
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(sample_defaults["Tech Support"]))
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(sample_defaults["Streaming TV"]))
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(sample_defaults["Streaming Movies"]))
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(sample_defaults["Contract"]))
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(sample_defaults["Paperless Billing"]))
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(sample_defaults["Payment Method"])
    )
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=float(sample_defaults["Monthly Charges"]))
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=float(sample_defaults["Total Charges"]))

    predict_btn = st.button("Predict Customer Status")

input_data = pd.DataFrame([{
    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure_months,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges
}])

if not predict_btn:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.04); padding: 1.2rem; border-radius: 18px; border: 1px solid rgba(255,255,255,0.08);">
        <h3 style="color:white;">Welcome</h3>
        <p style="color:#cbd5e1;">
            Use the left sidebar to enter customer details or load a sample customer profile.
            Then click <b>Predict Customer Status</b> to view leaving probability, risk level, key reasons, strategy, and report download.
        </p>
    </div>
    """, unsafe_allow_html=True)

if predict_btn:
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    customer_status = "Will Leave" if prediction == 1 else "Will Stay"
    leaving_probability = probability * 100
    risk_level = get_risk_level(probability)
    confidence_score = get_confidence_score(probability)
    top_reasons = get_top_reasons(input_data.iloc[0])
    retention_strategies = get_retention_strategy(input_data.iloc[0])

    risk_class = "low-risk"
    if risk_level == "Medium":
        risk_class = "medium-risk"
    elif risk_level == "High":
        risk_class = "high-risk"

    st.subheader("Prediction Results")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Customer Status</div>
            <div class="metric-value">{customer_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Leaving Probability</div>
            <div class="metric-value">{leaving_probability:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value">{risk_level}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence Score</div>
            <div class="metric-value">{confidence_score:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.progress(min(int(leaving_probability), 100))

    if prediction == 1:
        st.error("This customer is likely to leave.")
    else:
        st.success("This customer is likely to stay.")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Top Reasons")
        for reason in top_reasons:
            st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

        st.subheader("Recommended Retention Strategy")
        for strategy in retention_strategies:
            st.markdown(f'<div class="strategy-box">{strategy}</div>', unsafe_allow_html=True)

    with right:
        st.subheader("Leaving Probability Chart")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=leaving_probability,
            number={'suffix': "%"},
            title={'text': "Leaving Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 35], 'color': "#065f46"},
                    {'range': [35, 70], 'color': "#92400e"},
                    {'range': [70, 100], 'color': "#991b1b"}
                ]
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white", "size": 16},
            margin=dict(l=20, r=20, t=50, b=20),
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    report_data = pd.DataFrame({
        "Metric": [
            "Customer Status",
            "Leaving Probability (%)",
            "Risk Level",
            "Confidence Score (%)",
            "Top Reason 1",
            "Top Reason 2",
            "Top Reason 3",
            "Retention Strategy 1",
            "Retention Strategy 2",
            "Retention Strategy 3"
        ],
        "Value": [
            customer_status,
            round(leaving_probability, 2),
            risk_level,
            round(confidence_score, 2),
            top_reasons[0] if len(top_reasons) > 0 else "",
            top_reasons[1] if len(top_reasons) > 1 else "",
            top_reasons[2] if len(top_reasons) > 2 else "",
            retention_strategies[0] if len(retention_strategies) > 0 else "",
            retention_strategies[1] if len(retention_strategies) > 1 else "",
            retention_strategies[2] if len(retention_strategies) > 2 else ""
        ]
    })

    csv_buffer = io.StringIO()
    report_data.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download Prediction Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name="customer_leaving_report.csv",
        mime="text/csv"
    )