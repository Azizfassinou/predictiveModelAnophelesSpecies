import streamlit as st
import joblib
import re
import os

MODEL_PATH = 'M_Final.pkl'
VEC_PATH = 'V_Final.pkl'

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        st.error("Fichiers M_Final.pkl ou V_Final.pkl manquants.")
        return None, None
    model = joblib.load(MODEL_PATH)
    cv = joblib.load(VEC_PATH)
    return model, cv

model, cv = load_models()

st.set_page_config(page_title="Prédicteur d'Espèces d'Anopheles", page_icon="🧬")

st.title("🧬 Testeur d'Espèces Anopheles")
st.markdown("""
Cette application utilise un modèle de **Machine Learning** (Random Forest) entraîné sur des séquences ADN du gène COI.
""")

sequence_test = st.text_area("Collez la séquence ADN (format texte ou FASTA)", height=200)

if st.button("Prédire l'espèce"):
    if sequence_test:
        if sequence_test.startswith(">"):
            sequence_test = "\n".join(sequence_test.split("\n")[1:])

        dna_clean = re.sub(r'[^ATGCatgc]', '', sequence_test)
        dna_clean = dna_clean.upper()

        if len(dna_clean) < 100:
            st.error(f"Séquence trop courte ({len(dna_clean)} bp). Minimum 100 bp requis pour une prédiction fiable.")
        else:
            k = 6
            kmers = " ".join([dna_clean[i:i+k] for i in range(len(dna_clean) - k + 1)])

            data = cv.transform([kmers])
            resultat = model.predict(data)[0]
            proba_list = model.predict_proba(data)[0]

            class_idx = list(model.classes_).index(resultat)
            confiance = proba_list[class_idx]

            st.divider()
            if confiance < 0.70:
                st.warning(f" **Attention : Certitude faible ({confiance*100:.1f}%)**")
                st.info(f"Le modèle hésite, mais l'espèce la plus probable est : **{resultat}**")
            else:
                st.success(f"Espèce identifiée : **{resultat}**")
                st.metric("Indice de confiance", f"{confiance*100:.1f}%")

            with st.expander("Détails de la séquence analysée"):
                st.write(f"Longueur : {len(dna_clean)} paires de bases")
                st.code(dna_clean[:100] + "...")
    else:
        st.error("Veuillez entrer une séquence ADN.")
