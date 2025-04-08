import numpy as np
import pandas as pd
import joblib
import gradio as gr
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load model, scaler, and metadata
bundle = joblib.load("factor_weight_model.pkl")
model = bundle['model']
scaler = bundle['scaler']
a10_max_value = bundle['a10_max_value']
feature_names = bundle['feature_names']
X_train_scaled = bundle['X_train_scaled']

# LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    mode='regression'
)

# Predict function with LIME
def predict_with_lime(A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12):
    A10_corrected = a10_max_value - A10
    input_df = pd.DataFrame([[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10_corrected, A11, A12]], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    # Generate LIME explanation
    exp = explainer.explain_instance(input_scaled[0], model.predict, num_features=5)
    fig = exp.as_pyplot_figure()
    fig.savefig("lime_explanation.png")

    return round(pred), "lime_explanation.png"

# Gradio interface
iface = gr.Interface(
    fn=predict_with_lime,
    inputs=[
        gr.Number(label="A1"),
        gr.Number(label="A2"),
        gr.Number(label="A3 (%)"),
        gr.Number(label="A4 (%)"),
        gr.Number(label="A5 (Dollar Amount)"),
        gr.Number(label="A6 (Dollar Amount)"),
        gr.Number(label="A7"),
        gr.Number(label="A8"),
        gr.Number(label="A9"),
        gr.Number(label="A10 (%) - Lower is Better"),
        gr.Number(label="A11 (%)"),
        gr.Number(label="A12"),
    ],
    outputs=[
        gr.Number(label="Predicted Score"),
        gr.Image(type="filepath", label="LIME Explanation")
    ],
    title="Factor Weight Predictor with LIME",
    description="Enter values to predict the overall factor weight. LIME explains top 5 features influencing the prediction."
)

iface.launch()
