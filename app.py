import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Ouaga Smart City - Alerte Harmattan",
    page_icon="🌍",
    layout="wide"
)

@st.cache_resource
def charger_modeles():
    rf_reg = joblib.load('models/rf_regression.pkl')
    rf_clf = joblib.load('models/rf_classification.pkl')
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    return rf_reg, rf_clf, config

@st.cache_data
def charger_donnees():
    df = pd.read_csv('data/ouaga_dataset_final_ml.csv')
    return df

rf_reg, rf_clf, config = charger_modeles()
df = charger_donnees()
seuil_optimal = config['seuil_optimal']
features      = config['features']

st.sidebar.title("🌍 Ouaga Smart City")
st.sidebar.markdown("**Système d Alerte Précoce Harmattan**")
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Prédictions", "Analyse Historique", "Interprétabilité"]
)

if page == "Accueil":
    st.title("🌍 Prédiction du Stress Environnemental")
    st.subheader("Système d Alerte Précoce à Ouagadougou")

    st.markdown("""
    Ce dashboard prédit les épisodes de **Harmattan dangereux**
    3 jours à l avance à Ouagadougou (Burkina Faso)
    par fusion de données satellitaires multi-sources.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Période", "2018-2025")
    col2.metric("Observations", "2 735 jours")
    col3.metric("AUC-ROC", "0.799")
    col4.metric("Recall Dangereux", "71.7%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📡 Sources de Données")
        st.markdown("""
        | Source | Variable |
        |---|---|
        | Sentinel-5P | AAI Harmattan |
        | MODIS | Température Sol |
        | ERA5 | Météo complète |
        """)

    with col2:
        st.markdown("### 🤖 Modèles ML")
        st.markdown("""
        | Tâche | Modèle | Score |
        |---|---|---|
        | Régression AAI J+3 | Random Forest | R²=0.281 |
        | Classification J+3 | RF+SMOTE | AUC=0.799 |
        """)

    st.markdown("---")
    st.markdown("### 📍 Zone d Étude : Ouagadougou, Burkina Faso")

    fig = go.Figure(go.Scattermapbox(
        lat=[12.3647],
        lon=[-1.5144],
        mode='markers+text',
        marker=dict(size=20, color='red'),
        text=['Ouagadougou'],
        textposition='top right'
    ))
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=12.3647, lon=-1.5144),
            zoom=10
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Prédictions":
    st.title("🔮 Prédictions AAI J+3")
    st.markdown("Entrez les valeurs du jour pour prédire dans 3 jours.")

    st.markdown("### Valeurs du jour")

    col1, col2, col3 = st.columns(3)

    with col1:
        aai_lag1 = st.number_input("AAI hier (lag1)", -3.0, 6.0, 0.2, 0.1)
        aai_lag2 = st.number_input("AAI avant-hier (lag2)", -3.0, 6.0, 0.1, 0.1)
        aai_lag3 = st.number_input("AAI il y a 3j (lag3)", -3.0, 6.0, 0.1, 0.1)
        aai_roll7 = st.number_input("Moyenne AAI 7j", -3.0, 6.0, 0.2, 0.1)
        aai_roll14 = st.number_input("Moyenne AAI 14j", -3.0, 6.0, 0.2, 0.1)
        aai_roll30 = st.number_input("Moyenne AAI 30j", -3.0, 6.0, 0.2, 0.1)

    with col2:
        temperature = st.number_input("Température (°C)", 15.0, 45.0, 28.0, 0.5)
        lst = st.number_input("LST Température Sol (°C)", 15.0, 55.0, 35.0, 0.5)
        precipitation = st.number_input("Précipitation (mm)", 0.0, 100.0, 0.0, 0.5)
        humidite = st.number_input("Humidité", -30.0, 30.0, 0.0, 0.5)
        pression = st.number_input("Pression (hPa)", 950.0, 1010.0, 977.0, 0.5)
        vitesse_vent = st.number_input("Vitesse Vent (m/s)", 0.0, 15.0, 3.0, 0.1)

    with col3:
        no2 = st.number_input("NO2 (µmol/m²)", 0.0, 150.0, 60.0, 1.0)
        co = st.number_input("CO (mol/m²)", 0.0, 0.1, 0.036, 0.001)
        mois_sin = st.number_input("Mois Sin", -1.0, 1.0, 0.5, 0.1)
        mois_cos = st.number_input("Mois Cos", -1.0, 1.0, 0.5, 0.1)
        jour_sin = st.number_input("Jour Sin", -1.0, 1.0, 0.5, 0.1)
        saison_seche = st.selectbox("Saison Sèche", [0, 1])

    if st.button("🔮 Prédire AAI J+3", type="primary"):
        input_data = {f: 0 for f in features}
        input_data.update({
            'AAI_lag1': aai_lag1, 'AAI_lag2': aai_lag2,
            'AAI_lag3': aai_lag3, 'AAI_roll7': aai_roll7,
            'AAI_roll14': aai_roll14, 'AAI_roll30': aai_roll30,
            'temperature': temperature, 'LST': lst,
            'precipitation': precipitation, 'humidite': humidite,
            'pression': pression, 'vitesse_vent': vitesse_vent,
            'NO2': no2, 'CO': co, 'mois_sin': mois_sin,
            'mois_cos': mois_cos, 'jour_sin': jour_sin,
            'saison_seche': saison_seche
        })

        X_input   = pd.DataFrame([input_data])[features]
        pred_reg  = rf_reg.predict(X_input)[0]
        pred_prob = rf_clf.predict_proba(X_input)[0][1]
        pred_clf  = int(pred_prob >= seuil_optimal)

        st.markdown("---")
        st.markdown("### Résultats Prédiction J+3")

        col1, col2, col3 = st.columns(3)
        col1.metric("AAI Prédit J+3", f"{pred_reg:.3f}")
        col2.metric("Probabilité Dangereux", f"{pred_prob*100:.1f}%")

        if pred_clf == 1:
            col3.error("🔴 ALERTE HARMATTAN !")
            st.error("""
            ⚠️ Episode Harmattan dangereux prévu dans 3 jours !
            Mesures recommandées :
            - Porter masques de protection
            - Limiter activités extérieures
            - Alerter établissements scolaires
            """)
        else:
            col3.success("🟢 Qualité Air Normale")
            st.success("✅ Aucun épisode Harmattan dangereux prévu dans 3 jours.")

        if pred_reg < 0:
            niveau = "🟢 Faible"
        elif pred_reg < 1:
            niveau = "🟡 Modéré"
        elif pred_reg < 2:
            niveau = "🟠 Élevé"
        else:
            niveau = "🔴 Critique"

        st.info(f"Niveau de risque AAI : {niveau}")

elif page == "Analyse Historique":
    st.title("📊 Analyse Historique AAI 2018-2025")

    st.markdown("### Evolution AAI journalier")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df['AAI'], mode='lines',
        name='AAI journalier',
        line=dict(color='steelblue', width=0.8)
    ))
    fig.add_hline(y=1, line_dash='dash', line_color='orange',
                  annotation_text='Seuil Élevé')
    fig.add_hline(y=2, line_dash='dash', line_color='red',
                  annotation_text='Seuil Critique')
    fig.update_layout(
        title='Evolution AAI Ouagadougou 2018-2025',
        xaxis_title='Jours', yaxis_title='AAI',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distribution des niveaux AAI")
    col1, col2 = st.columns(2)

    with col1:
        labels = ['Faible\n(AAI<0)', 'Modéré\n(0-1)',
                  'Élevé\n(1-2)', 'Critique\n(>2)']
        sizes  = [
            (df['AAI'] < 0).sum(),
            ((df['AAI'] >= 0) & (df['AAI'] < 1)).sum(),
            ((df['AAI'] >= 1) & (df['AAI'] < 2)).sum(),
            (df['AAI'] >= 2).sum()
        ]
        colors = ['green', 'yellow', 'orange', 'red']
        fig2   = go.Figure(go.Pie(
            labels=labels, values=sizes,
            marker=dict(colors=colors)
        ))
        fig2.update_layout(title='Distribution Niveaux AAI')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### Statistiques AAI")
        st.dataframe(df['AAI'].describe().round(3))

elif page == "Interprétabilité":
    st.title("🔍 Interprétabilité du Modèle")

    st.markdown("### Variables les plus importantes (SHAP)")
    st.markdown("""
    Les SHAP values expliquent la contribution de chaque variable
    aux prédictions du modèle.
    """)

    shap_data = {
        'Feature': ['AAI_roll14', 'AAI_roll30', 'AAI_lag2',
                    'AAI_lag30', 'AAI_lag3', 'AAI_lag7',
                    'LST_lag7', 'AAI_lag14', 'LST', 'pression'],
        'Importance': [0.205, 0.126, 0.119, 0.044, 0.040,
                       0.035, 0.027, 0.026, 0.023, 0.018]
    }
    df_shap = pd.DataFrame(shap_data).sort_values(
        'Importance', ascending=True
    )

    fig = go.Figure(go.Bar(
        x=df_shap['Importance'],
        y=df_shap['Feature'],
        orientation='h',
        marker=dict(color='steelblue')
    ))
    fig.update_layout(
        title='Top 10 Variables Importantes - Régression AAI J+3',
        xaxis_title='Importance SHAP',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interprétation physique")
    st.markdown("""
    | Variable | Importance | Signification |
    |---|---|---|
    | AAI_roll14 | 20.5% | Tendance Harmattan 14 derniers jours |
    | AAI_roll30 | 12.6% | Tendance long terme 30 jours |
    | AAI_lag2 | 11.9% | Signal avant-hier |
    | LST_lag7 | 2.7% | Chaleur semaine passée |
    | pression | 1.8% | Signal météorologique |

    > **Conclusion** : Les moyennes glissantes dominent les prédictions.
    Le Harmattan est un phénomène persistant : si les 2 dernières semaines
    étaient poussiéreuses, les prochains jours le seront aussi.
    """)

    st.markdown("### Métriques finales des modèles")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Régression AAI J+3**")
        st.metric("RMSE", "0.678")
        st.metric("MAE", "0.508")
        st.metric("R²", "0.281")

    with col2:
        st.markdown("**Classification Harmattan J+3**")
        st.metric("AUC-ROC", "0.799")
        st.metric("Recall", "71.7%")
        st.metric("F1 Score", "0.487")

st.sidebar.markdown("---")
st.sidebar.markdown("**Projet Smart City**")
st.sidebar.markdown("Formation Data Scientist")
st.sidebar.markdown("Ouagadougou, Burkina Faso")