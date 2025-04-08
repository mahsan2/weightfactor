# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import lime
# import lime.lime_tabular
# import matplotlib.pyplot as plt
# import os

# # Load model and data
# bundle = joblib.load("factor_weight_model.pkl")
# model = bundle["model"]
# scaler = bundle["scaler"]
# a10_max_value = bundle["a10_max_value"]
# feature_names = bundle["feature_names"]
# X_train_scaled = bundle["X_train_scaled"]

# # LIME Explainer
# explainer = lime.lime_tabular.LimeTabularExplainer(
#     training_data=X_train_scaled,
#     feature_names=feature_names,
#     mode="regression"
# )

# st.title("Factor Weight Prediction with LIME Explanation")

# # User Inputs
# inputs = {}
# for i, label in enumerate([
#     "A1", "A2", "A3 (%)", "A4 (%)",
#     "A5 ($)", "A6 ($)", "A7", "A8", "A9",
#     "A10 (%) - lower is better", "A11 (%)", "A12"
# ]):
#     key = f"A{i+1}"
#     value = st.number_input(label, value=0.0)
#     inputs[key] = value

# if st.button("Predict and Explain"):
#     input_df = pd.DataFrame([inputs])
#     input_df["A10"] = a10_max_value - input_df["A10"]  # correct A10
#     input_scaled = scaler.transform(input_df)

#     pred = model.predict(input_scaled)[0]
#     st.success(f"Predicted Score: {round(pred)}")

#     # LIME
#     exp = explainer.explain_instance(input_scaled[0], model.predict, num_features=5)
#     fig = exp.as_pyplot_figure()
#     fig_path = "lime_explanation.png"
#     fig.savefig(fig_path)
#     st.image(fig_path, caption="LIME Explanation")
#     os.remove(fig_path)  # clean up

#############################################
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
import streamlit_authenticator as stauth

# ---- Auth Setup ----
names = ['Zahed']
usernames = ['zahed1']
hashed_passwords = ['$2b$12$F7VEhfTFC24VcsUTOfBBm.oYnimMYxqIi9g1EVT6L/liSbqZPX3gu']  # Replace with your generated hash

authenticator = stauth.Authenticate(
    names=names,
    usernames=usernames,
    passwords=hashed_passwords,
    cookie_name="auth_cookie",
    key="abcdef",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login('Login', location='main')

if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.title("Factor Weight Prediction with LIME Explanation")

    # Load model and data
    bundle = joblib.load("factor_weight_model.pkl")
    model = bundle["model"]
    scaler = bundle["scaler"]
    a10_max_value = bundle["a10_max_value"]
    feature_names = bundle["feature_names"]
    X_train_scaled = bundle["X_train_scaled"]

    # LIME Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        mode="regression"
    )

    # User Inputs
    inputs = {}
    for i, label in enumerate([
        "A1", "A2", "A3 (%)", "A4 (%)",
        "A5 ($)", "A6 ($)", "A7", "A8", "A9",
        "A10 (%) - lower is better", "A11 (%)", "A12"
    ]):
        key = f"A{i+1}"
        value = st.number_input(label, value=0.0)
        inputs[key] = value

    if st.button("Predict and Explain"):
        input_df = pd.DataFrame([inputs])
        input_df["A10"] = a10_max_value - input_df["A10"]  # correct A10
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        st.success(f"Predicted Score: {round(pred)}")

        # LIME
        exp = explainer.explain_instance(input_scaled[0], model.predict, num_features=5)
        fig = exp.as_pyplot_figure()
        fig_path = "lime_explanation.png"
        fig.savefig(fig_path)
        st.image(fig_path, caption="LIME Explanation")
        os.remove(fig_path)  # clean up

elif authentication_status is False:
    st.error("Invalid username or password")
elif authentication_status is None:
    st.warning("Please enter your credentials")
