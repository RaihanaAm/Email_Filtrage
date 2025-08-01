
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Spam",
    page_icon="📧",
    layout="wide"
)

# Titre principal
st.title("🚨 Détecteur de Spam Email")
st.markdown("---")

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    try:
        # Essayer de charger le pipeline d'abord
        model = joblib.load('model_pipeline.pkl')
        return model, "pipeline"
    except:
        try:
            # Sinon charger modèle et vectoriseur séparément
            model = joblib.load('model.pkl')
            vectorizer = joblib.load('vectorizer.pkl')
            return (model, vectorizer), "separate"
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {e}")
            return None, None

# Fonction de prédiction
def predict_spam(text, model_data, model_type):
    try:
        if model_type == "pipeline":
            prediction = model_data.predict([text])
            proba = model_data.predict_proba([text])
        else:
            model, vectorizer = model_data
            text_vectorized = vectorizer.transform([text])
            prediction = model.predict(text_vectorized)
            proba = model.predict_proba(text_vectorized)
        
        return prediction[0], proba[0]
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None, None

# Charger le modèle
model_data, model_type = load_model()



# Interface principale
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Entrez votre email à analyser")
    
    # Zone de texte pour l'email
    email_text = st.text_area(
        "Texte de l'email:",
        height=200,
        placeholder="Collez ici le contenu de l'email à analyser..."
    )
    
    # Bouton de prédiction
    if st.button("🔍 Analyser l'email", type="primary"):
        if email_text.strip():
            with st.spinner("Analyse en cours..."):
                prediction, probabilities = predict_spam(email_text, model_data, model_type)
                
                if prediction is not None:
                    # Affichage du résultat
                    if prediction == 1:
                        st.error("🚨 **SPAM DÉTECTÉ**")
                        confidence = probabilities[1] * 100
                    else:
                        st.success("✅ **EMAIL LÉGITIME**")
                        confidence = probabilities[0] * 100
                    
                    # Barre de progression pour la confiance
                    st.metric("Niveau de confiance", f"{confidence:.1f}%")
                    st.progress(confidence/100)
        else:
            st.warning("⚠️ Veuillez entrer un texte à analyser")

with col2:
    st.subheader("📊 Informations")
    
    # Statistiques du modèle
    st.info(f"**Type de modèle:** {model_type}")
    st.info(f"**Dernière analyse:** {datetime.now().strftime('%H:%M:%S')}")
    
    # Exemples de spam
    st.subheader("🔍 Exemples de test")
    
    examples = [
        "Félicitations ! Vous avez gagné 1000€ ! Cliquez ici maintenant !",
        "Réunion prévue demain à 14h en salle de conférence.",
        "URGENT ! Votre compte sera suspendu ! Confirmez vos données bancaires !"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"Tester exemple {i+1}", key=f"example_{i}"):
            prediction, probabilities = predict_spam(example, model_data, model_type)
            if prediction == 1:
                st.error(f"SPAM ({probabilities[1]*100:.1f}%)")
            else:
                st.success(f"LÉGITIME ({probabilities[0]*100:.1f}%)")

# Section historique (optionnel)
st.markdown("---")
st.subheader("📈 Historique des analyses")

# Initialiser l'historique dans la session
if 'history' not in st.session_state:
    st.session_state.history = []

# Ajouter à l'historique quand une prédiction est faite
if email_text and st.button("Sauvegarder dans l'historique"):
    if email_text.strip():
        prediction, probabilities = predict_spam(email_text, model_data, model_type)
        if prediction is not None:
            entry = {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Email (extrait)': email_text[:50] + "..." if len(email_text) > 50 else email_text,
                'Résultat': 'SPAM' if prediction == 1 else 'LÉGITIME',
                'Confiance': f"{(probabilities[1] if prediction == 1 else probabilities[0])*100:.1f}%"
            }
            st.session_state.history.append(entry)

# Afficher l'historique
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    
    if st.button("🗑️ Vider l'historique"):
        st.session_state.history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Détecteur de spam basé sur SVM - Développé avec Streamlit*")
