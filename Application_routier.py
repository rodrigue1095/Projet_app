1
import streamlit as st
import pickle
import numpy as np

with open("classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

st.title("Application Du Traffic Routier_nante")
st.write("Entrer les valeur du traffic routier.")

cha_long = st.number_input("Longueur", min_value=45, max_value=2004)
mf1_debit = st.number_input("Debit", min_value=-1, max_value=9120)
mf1_vit = st.number_input("Vitesse", min_value=-1, max_value=150)
mf1_taux = st.number_input(" Taux Occupation", min_value=-1, max_value=100)
tp_couleur = st.number_input(" Code couleur", min_value=2, max_value=6)
tc1_temps = st.number_input(" Temps de parcours", min_value=-1, max_value=30268)

if st.button("Predict"):
    features = np.array([[cha_long, mf1_debit, mf1_vit, mf1_taux,tp_couleur,tc1_temps]])
    prediction = model.predict(features)
    st.write(f"Etats Du Traffic_nante : {prediction[0]}")