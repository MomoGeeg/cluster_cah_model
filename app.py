import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st


# Configuration de la page
st.set_page_config(
    page_title="Segmentation des clients - modèle CAH",
    page_icon="📊",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de clustering (CAH - classification ascendante hiérarchique) "
    "permet de déterminer à quelle classe appartient un client à partir de "
    "ses caractéristiques d'utilisation."
)


# Chargement des artefacts (mis en cache : chargés une seule fois)

@st.cache_resource
def load_artifacts():
    artefacts = jb.load("modele_cah.joblib")
    return artefacts


artefacts = load_artifacts()
centroides = np.array(artefacts["centroides"])
scaler = artefacts["scaler"]
colonnes = artefacts["colonnes"]
noms_classes = artefacts["noms_classes"]


# Fonction de prédiction simple

def pred_func(valeurs):
    client = np.array(valeurs, dtype=float).reshape(1, -1)
    client = scaler.transform(client)

    distances = np.linalg.norm(centroides - client, axis=1)
    classe_predite = distances.argmin()

    nom_classe = noms_classes.get(classe_predite, f"Classe {classe_predite}")
    return nom_classe


# Fonction de prédiction multiple

def pred_func_csv(file):
    df = pd.read_csv(file)
    prediction = []

    for i, row in enumerate(df[colonnes].values):
        try:
            resultat = pred_func(row)
            prediction.append(resultat)
        except Exception as e:
            print(f"Erreur ligne {i} : {row} → {e}")
            prediction.append(None)

    df['classe_predite'] = prediction
    return df


# Interface

st.title("📊 Segmentation des clients - modèle CAH")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire la classe d'un client avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        valeurs_saisies = []
        cols = st.columns(2)
        for i, col_nom in enumerate(colonnes):
            with cols[i % 2]:
                valeur = st.number_input(col_nom, value=0.0, format="%.2f")
                valeurs_saisies.append(valeur)

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = pred_func(valeurs_saisies)
            st.success(f"**Ce client appartient à :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire la classe de plusieurs clients")
    st.write(DESCRIPTION)
    st.caption(f"Le fichier CSV doit contenir, dans cet ordre, les colonnes : {', '.join(colonnes)}.")

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")