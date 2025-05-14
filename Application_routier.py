# Charger les bibrary
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Importer la base de données et faire une description
df = pd.read_csv("Fluidite_routiers_nantes_0.csv")
df.drop('id', axis=1, inplace=True)
df.head(5)
df.drop('cha_id', axis=1, inplace=True)
df.drop('cha_lib', axis=1, inplace=True)
df.drop('mf1_hd', axis=1, inplace=True)
df.drop('geometry', axis=1, inplace=True)
df.drop('lon', axis=1, inplace=True)
df.drop('lat', axis=1, inplace=True)
df.head(5)
X = df.drop('etat_trafic', axis=1)
y = df['etat_trafic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

votre_model = RandomForestClassifier()
#model.fit(X_train, y_train)
model = votre_model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

import streamlit as st
import pickle
import numpy as np

with open("modele.pkl", "wb") as f:
    pickle.dump(model, f)

try:
    with open("modele.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except EOFError:
    st.error("Erreur : le fichier de modèle est vide ou corrompu.")
    
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