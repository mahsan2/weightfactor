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
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import lime
# import lime.lime_tabular
# import matplotlib.pyplot as plt
# import os
# import streamlit_authenticator as stauth

# # --- FIXED AUTH SECTION (compatible with streamlit-authenticator >=0.2.3) ---

# # Credentials dictionary
# credentials = {
#     "usernames": {
#         "zahed1": {
#             "name": "Zahed",
#             "password": "$2b$12$F7VEhfTFC24VcsUTOfBBm.oYnimMYxqIi9g1EVT6L/liSbqZPX3gu"  # hashed password for OUexplain834!
#         }
#     }
# }

# # Initialize authenticator (note the updated constructor format)
# authenticator = stauth.Authenticate(
#     credentials,
#     cookie_name="auth_cookie",
#     key="abcdef",
#     cookie_expiry_days=1
# )

# # --- LOGIN INTERFACE ---
# name, authentication_status, username = authenticator.login("Login", location="main")

# if authentication_status:
#     authenticator.logout("Logout", location="sidebar")
#     st.title("Factor Weight Prediction with LIME Explanation")

#     # Load model and data
#     bundle = joblib.load("factor_weight_model.pkl")
#     model = bundle["model"]
#     scaler = bundle["scaler"]
#     a10_max_value = bundle["a10_max_value"]
#     feature_names = bundle["feature_names"]
#     X_train_scaled = bundle["X_train_scaled"]

#     # LIME Explainer
#     explainer = lime.lime_tabular.LimeTabularExplainer(
#         training_data=X_train_scaled,
#         feature_names=feature_names,
#         mode="regression"
#     )

#     # User Inputs
#     inputs = {}
#     for i, label in enumerate([
#         "A1", "A2", "A3 (%)", "A4 (%)",
#         "A5 ($)", "A6 ($)", "A7", "A8", "A9",
#         "A10 (%) - lower is better", "A11 (%)", "A12"
#     ]):
#         key = f"A{i+1}"
#         value = st.number_input(label, value=0.0)
#         inputs[key] = value

#     if st.button("Predict and Explain"):
#         input_df = pd.DataFrame([inputs])
#         input_df["A10"] = a10_max_value - input_df["A10"]  # correct A10
#         input_scaled = scaler.transform(input_df)

#         pred = model.predict(input_scaled)[0]
#         st.success(f"Predicted Score: {round(pred)}")

#         # LIME
#         exp = explainer.explain_instance(input_scaled[0], model.predict, num_features=5)
#         fig = exp.as_pyplot_figure()
#         fig.tight_layout()
#         fig_path = "lime_explanation.png"
#         fig.savefig(fig_path)
#         st.image(fig_path, caption="LIME Explanation")
#         os.remove(fig_path)

# elif authentication_status is False:
#     st.error("Invalid username or password")
# elif authentication_status is None:
#     st.warning("Please enter your credentials")
########################################################################
###########################################################################
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import lime
# import lime.lime_tabular
# import matplotlib.pyplot as plt
# import os
# import streamlit_authenticator as stauth

# # --- FIXED AUTH SECTION (compatible with streamlit-authenticator >=0.2.3) ---

# # Credentials dictionary
# credentials = {
#     "usernames": {
#         "zahed1": {
#             "name": "Zahed",
#             "password": "$2b$12$F7VEhfTFC24VcsUTOfBBm.oYnimMYxqIi9g1EVT6L/liSbqZPX3gu"  # hashed password for OUexplain834!
#         }
#     }
# }

# # Initialize authenticator
# authenticator = stauth.Authenticate(
#     credentials,
#     cookie_name="auth_cookie",
#     key="abcdef",
#     cookie_expiry_days=1
# )

# # --- LOGIN INTERFACE ---
# name, authentication_status, username = authenticator.login("Login", location="main")

# if authentication_status:
#     authenticator.logout("Logout", location="sidebar")
#     st.title("Factor Weight Prediction with LIME Explanation")

#     # Load model and data
#     bundle = joblib.load("factor_weight_model.pkl")
#     model = bundle["model"]
#     scaler = bundle["scaler"]
#     a10_max_value = bundle["a10_max_value"]
#     feature_names = bundle["feature_names"]
#     X_train_scaled = bundle["X_train_scaled"]

#     # LIME Explainer
#     explainer = lime.lime_tabular.LimeTabularExplainer(
#         training_data=X_train_scaled,
#         feature_names=feature_names,
#         mode="regression"
#     )

#     # User Inputs
#     inputs = {}
#     for i, label in enumerate([
#         "A1", "A2", "A3 (%)", "A4 (%)",
#         "A5 ($)", "A6 ($)", "A7", "A8", "A9",
#         "A10 (%) - lower is better", "A11 (%)", "A12"
#     ]):
#         key = f"A{i+1}"
#         value = st.number_input(label, value=0.0)
#         inputs[key] = value

#     if st.button("Predict and Explain"):
#         input_df = pd.DataFrame([inputs])
#         input_df["A10"] = a10_max_value - input_df["A10"]  # correct A10
#         input_scaled = scaler.transform(input_df)

#         pred = model.predict(input_scaled)[0]
#         st.success(f"Predicted Score: {round(pred)}")

#         # LIME Plot
#         exp = explainer.explain_instance(input_scaled[0], model.predict, num_features=5)
#         fig = exp.as_pyplot_figure()
#         fig.tight_layout()
#         fig_path = "lime_explanation.png"
#         fig.savefig(fig_path)
#         st.image(fig_path, caption="LIME Explanation")
#         os.remove(fig_path)

#         # Real names for features
#         feature_map = {
#             "A1": "Citations Per Publication",
#             "A2": "Field Weighted Citation Impact",
#             "A3": "Publications Cited in Top 5% of Journals",
#             "A4": "Publications Cited in Top 25% of Journals",
#             "A5": "Research Expenditures (in millions)",
#             "A6": "Research Expenditures per Faculty (in thousands)",
#             "A7": "Peer Assessment Score",
#             "A8": "Recruiter Assessment Score",
#             "A9": "Doctoral Degrees Granted",
#             "A10": "Acceptance Rate (doctoral)",
#             "A11": "% Faculty in National Academy of Engineering",
#             "A12": "Doctoral Student/Faculty Ratio"
#         }

#         # Explanation & Suggestion
#         st.markdown("### 🔍 What contributed to the prediction")
#         explanation = ""
#         suggestion = ""

#         for feature, impact in exp.as_list():
#             feat_code = feature.split()[0]
#             readable = feature_map.get(feat_code, feat_code)
#             direction = "increased" if impact > 0 else "decreased"
#             importance = abs(impact)

#             explanation += f"**{feature}** → Your **{readable}** {direction} the predicted score.  \n"

#             if impact > 0:
#                 if importance > 5:
#                     suggestion += f"✅ Strong positive from **{readable}** → _Maintain or improve it._  \n"
#             else:
#                 if importance > 5:
#                     suggestion += f"❗⃣ Strong negative from **{readable}** → _Focus on improving it._  \n"
#                 else:
#                     suggestion += f"⚠️ Minor negative from **{readable}** → _Can be improved._  \n"

#         st.markdown("### 🧠 Explanation")
#         st.markdown(explanation)

#         st.markdown("### 📈 Suggestions for Improvement")
#         st.markdown(suggestion)

# elif authentication_status is False:
#     st.error("Invalid username or password")
# elif authentication_status is None:
#     st.warning("Please enter your credentials")

###############
################
#################
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
import streamlit_authenticator as stauth

# ---------------- AUTHENTICATION SETUP ----------------
credentials = {
    "usernames": {
        "zahed1": {
            "name": "Zahed",
            "password": "$2b$12$F7VEhfTFC24VcsUTOfBBm.oYnimMYxqIi9g1EVT6L/liSbqZPX3gu"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    cookie_name="auth_cookie",
    key="abcdef",
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login("Login", location="main")

if authentication_status:
    authenticator.logout("Logout", location="sidebar")
    st.title("Factor Weight Prediction with LIME Explanation")

    # ---------------- LOAD MODEL ----------------
    bundle = joblib.load("factor_weight_model.pkl")
    model = bundle["model"]
    scaler = bundle["scaler"]
    a10_max_value = bundle["a10_max_value"]
    feature_names = bundle["feature_names"]
    X_train_scaled = bundle["X_train_scaled"]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        mode="regression"
    )

    # ---------------- INPUT FIELDS ----------------
    inputs = {}
    input_labels = [
        "A1", "A2", "A3 (%)", "A4 (%)",
        "A5 ($)", "A6 ($)", "A7", "A8", "A9",
        "A10 (%) - lower is better", "A11 (%)", "A12"
    ]
    for i, label in enumerate(input_labels):
        key = f"A{i+1}"
        value = st.number_input(label, value=0.0)
        inputs[key] = value

    if st.button("Predict and Explain"):
        input_df = pd.DataFrame([inputs])
        input_df["A10"] = a10_max_value - input_df["A10"]
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        st.success(f"Predicted Score: {round(pred)}")

        # ---------------- LIME ----------------
        exp = explainer.explain_instance(input_scaled[0], model.predict, num_features=12)
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig_path = "lime_explanation.png"
        fig.savefig(fig_path)
        st.image(fig_path, caption="LIME Explanation")

        # ---------------- EXPLANATION ----------------
        feature_map = {
            "A1": "Citations Per Publication",
            "A2": "Field Weighted Citation Impact",
            "A3": "Publications Cited in Top 5% of Journals",
            "A4": "Publications Cited in Top 25% of Journals",
            "A5": "Research Expenditures (in millions)",
            "A6": "Research Expenditures per Faculty (in thousands)",
            "A7": "Peer Assessment Score",
            "A8": "Recruiter Assessment Score",
            "A9": "Doctoral Degrees Granted",
            "A10": "Acceptance Rate (doctoral)",
            "A11": "% Faculty in National Academy of Engineering",
            "A12": "Doctoral Student/Faculty Ratio"
        }

        explanation = ""
        improvement = ""

        for feature, impact in exp.as_list():
            feat_code = feature.split()[0]
            readable = feature_map.get(feat_code, feat_code)
            direction = "increased" if impact > 0 else "decreased"
            explanation += f"🔹 **{readable}** {direction} the predicted score. (Impact: {impact:.2f})\n"

            if impact > 0:
                if abs(impact) > 0.5:
                    improvement += f"✅ Maintain or enhance **{readable}**.\n"
            else:
                if abs(impact) > 0.5:
                    improvement += f"🔴 Improve **{readable}**. It negatively impacted the score.\n"
                else:
                    improvement += f"⚠️ **{readable}** had minor negative impact — optional to improve.\n"

        st.markdown("### 🧠 Explanation")
        st.markdown(explanation)

        st.markdown("### 📈 Suggestions for Improvement")
        st.markdown(improvement)

        # ---------------- REPORT PDF ----------------
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="University Factor Score Report", ln=True, align="C")
        pdf.ln(10)

        pdf.cell(200, 10, txt="Input Values:", ln=True)
        for k, v in inputs.items():
            pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)

        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Predicted Score: {round(pred)}", ln=True)

        pdf.ln(10)
        pdf.cell(200, 10, txt="Explanation:", ln=True)
        for line in explanation.split("\n"):
            pdf.cell(200, 10, txt=line, ln=True)

        pdf.ln(10)
        pdf.cell(200, 10, txt="Suggestions:", ln=True)
        for line in improvement.split("\n"):
            pdf.cell(200, 10, txt=line, ln=True)

        pdf.image(fig_path, w=170)
        pdf_path = "full_report.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Full Report", f, file_name="report.pdf", mime="application/pdf")

        os.remove(fig_path)
        os.remove(pdf_path)

elif authentication_status is False:
    st.error("Invalid username or password")
elif authentication_status is None:
    st.warning("Please enter your credentials")


