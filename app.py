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

# ---------------- AUTH SETUP ----------------
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

    input_labels = [
        "A1 - Citations Per Publication", "A2 - Field Weighted Citation Impact", "A3 - Publications Cited in Top 5% of Journals (%)",
        "A4 - Publications Cited in Top 25% of Journals (%)", "A5 - Research expenditures (in millions) ($)",
        "A6 - Research expenditures per faculty member ($)", "A7 - Peer assessment score",
        "A8 - Recruiter Assessment Score", "A9 - Doctoral degrees granted",
        "A10 - Acceptance rate (doctoral) (%) - lower is better",
        "A11 - % of full-time tenured or tenure-track faculty in National Academy of Engineering (%)",
        "A12 - Doctoral student/faculty ratio"
    ]

    inputs = {}
    for i, label in enumerate(input_labels):
        key = f"A{i+1}"
        value = st.number_input(label=label, key=key, value=0.0)
        inputs[key] = value

    if st.button("Predict and Explain") or st.session_state.get("predicted", False):
        st.session_state["predicted"] = True
        input_df = pd.DataFrame([inputs])
        input_df["A10"] = a10_max_value - input_df["A10"]
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        st.success(f"Predicted Score: {round(pred)}")

        exp = explainer.explain_instance(input_scaled[0], model.predict, num_features=12)
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig_path = "lime_explanation.png"
        fig.savefig(fig_path)
        st.image(fig_path, caption="LIME Explanation")

        # Paragraph explanation for Streamlit
        lime_paragraph = (
            "**Interpretation:** The figure above shows the contribution of each factor using LIME. "
            "Green bars indicate features that positively influenced the prediction, while red bars indicate "
            "features that had a negative effect on the predicted score."
        )
        st.markdown(lime_paragraph)

        # Prepare top 5 explanations and top 3 suggestions
        feature_map = {f"A{i+1}": label.split(" - ")[1] for i, label in enumerate(input_labels)}
        top_5 = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)[:5]

        explanation_lines = []
        suggestion_lines = []

        for idx, (feature, impact) in enumerate(top_5, start=1):
            feat_code = feature.split()[0]
            readable = feature_map.get(feat_code, feat_code)
            direction = "increased" if impact > 0 else "decreased"
            explanation_lines.append(f"{idx}. {readable} {direction} the predicted score.")

            if impact < 0:
                suggestion_lines.append(f"{idx}. Improve {readable}. It negatively impacted the score.")

        st.markdown("### 🔍 Explanation (Top 5 Features)")
        for line in explanation_lines:
            st.markdown(f"{line}")

        st.markdown("### 🧠 Suggestions for Improvement")
        for line in suggestion_lines[:3]:
            st.markdown(f"{line}")

        # ---------------- PDF REPORT ----------------
        class ReportPDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "University Factor Score Report", ln=True, align="C")
                self.ln(3)

            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "I", 10)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")

            def add_section(self, title):
                self.set_font("Arial", "B", 12)
                self.set_fill_color(240, 240, 240)
                self.cell(0, 10, title, ln=True, fill=True)
                self.ln(2)

            def add_table(self, data_dict):
                self.set_font("Arial", "B", 9)
                label_col_width = 160
                value_col_width = 30
                line_height = 6
                self.cell(label_col_width, line_height, "Factor", border=1)
                self.cell(value_col_width, line_height, "Input Value", border=1)
                self.ln(line_height)
                self.set_font("Arial", "", 8)

                for k, v in data_dict.items():
                    full_label = input_labels[int(k[1:]) - 1]
                    x_before = self.get_x()
                    y_before = self.get_y()
                    self.multi_cell(label_col_width, line_height, str(full_label), border=1)
                    self.set_xy(x_before + label_col_width, y_before)
                    self.multi_cell(value_col_width, line_height, str(v), border=1)
                    self.set_y(y_before + max(self.get_y() - y_before, line_height))

            def add_paragraph(self, lines):
                self.set_font("Arial", "", 11)
                for line in lines:
                    self.multi_cell(0, 8, line)
                    self.ln(0.5)

        def safe(text):
            return text.encode("latin-1", "replace").decode("latin-1")

        pdf = ReportPDF()
        pdf.add_page()
        pdf.add_section("Input Values")
        pdf.add_table(inputs)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Predicted Score: {round(pred)}", ln=True)

        pdf.image(fig_path, w=160)
        pdf.ln(4)
        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 8,
            "The figure above shows the contribution of each factor using LIME. "
            "Green bars indicate features that positively influenced the prediction, "
            "while red bars indicate features that had a negative effect on the predicted score.")

        pdf.add_section("Explanation (Top 5)")
        pdf.add_paragraph([safe(line) for line in explanation_lines])

        pdf.ln(2)
        pdf.add_section("Suggestions (Top 3)")
        pdf.add_paragraph([safe(line) for line in suggestion_lines[:3]])

        report_path = "university_score_report.pdf"
        pdf.output(report_path)

        with open(report_path, "rb") as f:
            st.download_button("📄 Download Full Report", f, file_name="University_Score_Report.pdf", mime="application/pdf")

elif authentication_status is False:
    st.error("Invalid username or password")
elif authentication_status is None:
    st.warning("Please enter your credentials")
