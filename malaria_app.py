import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("malaria_model.pkl")

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt


scaler = joblib.load("scaler.pkl")
model = joblib.load("malaria_model.pkl")

# Train model on synthetic data
def train_model():
    np.random.seed(42)
    n_samples = 200
    temperature = np.random.normal(37.5, 0.8, n_samples)
    nausea_level = np.random.randint(0, 6, n_samples)
    vomiting = np.random.choice([0, 1], n_samples)
    body_pain = np.random.randint(0, 6, n_samples)
    headache = np.random.randint(0, 6, n_samples)
    loss_of_appetite = np.random.randint(0, 6, n_samples)
    dizziness = np.random.choice([0, 1], n_samples)
    stomach_pain = np.random.choice([0, 1], n_samples)
    body_weakness = np.random.choice([0, 1], n_samples)

    malaria_score = (
        0.8 * temperature +
        0.5 * nausea_level +
        0.6 * vomiting +
        0.4 * body_pain +
        0.5 * headache +
        0.3 * loss_of_appetite +
        0.4 * dizziness +
        0.4 * stomach_pain +
        0.5 * body_weakness +
        np.random.normal(0, 1, n_samples)
    )

    data = pd.DataFrame({
        'Temperature': temperature,
        'Nausea_Level': nausea_level,
        'Vomiting': vomiting,
        'Body_Pain': body_pain,
        'Headache': headache,
        'Loss_of_Appetite': loss_of_appetite,
        'Dizziness': dizziness,
        'Stomach_Pain': stomach_pain,
        'Body_Weakness': body_weakness,
        'Malaria_Score': malaria_score
    })

    data['Malaria_Risk'] = pd.qcut(data['Malaria_Score'], q=3, labels=[0, 1, 2])
    X = data.drop(columns=['Malaria_Score', 'Malaria_Risk'])
    y = data['Malaria_Risk'].astype(int)

    #Ttrain scaler and mogel

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)


    # Save model and scaler
    joblib.dump(model, "malaria_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return model, scaler

    model, scaler = train_model()

# Streamlit UI
st.title("Malaria Risk Prediction App")

st.write("Enter patient symptoms to predict malaria risk level:")

temperature = st.slider("Temperature (°C)", 35.0, 41.0, 37.5)
nausea_level = st.slider("Nausea Level (0–5)", 0, 5, 2)
vomiting = st.selectbox("Vomiting", [0, 1])
body_pain = st.slider("Body Pain (0–5)", 0, 5, 2)
headache = st.slider("Headache (0–5)", 0, 5, 2)
loss_of_appetite = st.slider("Loss of Appetite (0–5)", 0, 5, 2)
dizziness = st.selectbox("Dizziness", [0, 1])
stomach_pain = st.selectbox("Stomach Pain", [0, 1])
body_weakness = st.selectbox("Body Weakness", [0, 1])

# Predict
feature_names = ['Temperature', 'Nausea_Level', 'Vomiting', 'Body_Pain', 'Headache',
                 'Loss_of_Appetite', 'Dizziness', 'Stomach_Pain', 'Body_Weakness']

input_data = np.array([[temperature, nausea_level, vomiting, body_pain, headache,
                        loss_of_appetite, dizziness, stomach_pain, body_weakness]])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]


risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
st.subheader("Predicted Malaria Risk:")
st.write(f"🩺 {risk_labels[prediction]}")


if submitted:
    input_data = pd.DataFrame([...])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    risk_levels = {0: "Low", 1: "Medium", 2: "High"}
    st.success(f"Predicted Malaria Risk: **{risk_levels[prediction]}**")

    #Radar chart of symptoms
    import matplotlib.pyplot as plt
    import numpy as np

    labels = input_data.columns.tolist()
    values = input_data.values.flatten().tolist()

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    st.subheader("Symptom Profile")
    st.pyplot(fig)
