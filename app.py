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
from sklearn.metrics import mean_squared_error, r2_score
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
df         = charger_donnees()
seuil      = config['seuil_optimal']
features   = config['features']

X     = df[features].dropna()
y_reg = df.loc[X.index, 'AAI_t3']
y_clf = df.loc[X.index, 'harmattan_bin_t3']

split_idx        = int(len(X) * 0.80)
X_test           = X.iloc[split_idx:]
y_reg_test       = y_reg.iloc[split_idx:]
y_clf_test       = y_clf.iloc[split_idx:]
mask             = y_reg_test.notna() & y_clf_test.notna()
X_test_clean     = X_test[mask]
y_reg_test_clean = y_reg_test[mask]
y_clf_test_clean = y_clf_test[mask]

y_pred_reg  = rf_reg.predict(X_test_clean)
y_proba_clf = rf_clf.predict_proba(X_test_clean)[:,1]
y_pred_clf  = (y_proba_clf >= seuil).astype(int)

st.sidebar.title("🌍 Ouaga Smart City")
st.sidebar.markdown("**Système d Alerte Précoce Harmattan**")
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Prédictions", "Analyse Historique",
     "Evaluation Modèles", "Interprétabilité"]
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
        st.markdown("### 🤖 Modèles Comparés")
        st.markdown("""
        | Modèle | R2 | AUC |
        |---|---|---|
        | Random Forest | 0.281 | 0.799 |
        | XGBoost | 0.249 | 0.797 |
        | LightGBM | 0.255 | 0.776 |
        | LSTM | 0.274 | - |
        | CNN-LSTM | 0.227 | - |
        """)

    st.markdown("---")
    st.markdown("### 📍 Zone d Étude : Ouagadougou")
    fig = go.Figure(go.Scattermapbox(
        lat=[12.3647], lon=[-1.5144],
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

    col1, col2, col3 = st.columns(3)

    with col1:
        aai_lag1  = st.number_input("AAI hier", -3.0, 6.0, 0.2, 0.1)
        aai_lag2  = st.number_input("AAI avant-hier", -3.0, 6.0, 0.1, 0.1)
        aai_lag3  = st.number_input("AAI il y a 3j", -3.0, 6.0, 0.1, 0.1)
        aai_roll7 = st.number_input("Moyenne AAI 7j", -3.0, 6.0, 0.2, 0.1)
        aai_roll14= st.number_input("Moyenne AAI 14j", -3.0, 6.0, 0.2, 0.1)
        aai_roll30= st.number_input("Moyenne AAI 30j", -3.0, 6.0, 0.2, 0.1)

    with col2:
        temperature  = st.number_input("Température (°C)", 15.0, 45.0, 28.0, 0.5)
        lst          = st.number_input("LST Sol (°C)", 15.0, 55.0, 35.0, 0.5)
        precipitation= st.number_input("Précipitation (mm)", 0.0, 100.0, 0.0, 0.5)
        humidite     = st.number_input("Humidité", -30.0, 30.0, 0.0, 0.5)
        pression     = st.number_input("Pression (hPa)", 950.0, 1010.0, 977.0, 0.5)
        vitesse_vent = st.number_input("Vitesse Vent (m/s)", 0.0, 15.0, 3.0, 0.1)

    with col3:
        no2          = st.number_input("NO2 (µmol/m²)", 0.0, 150.0, 60.0, 1.0)
        co           = st.number_input("CO (mol/m²)", 0.0, 0.1, 0.036, 0.001)
        mois_sin     = st.number_input("Mois Sin", -1.0, 1.0, 0.5, 0.1)
        mois_cos     = st.number_input("Mois Cos", -1.0, 1.0, 0.5, 0.1)
        jour_sin     = st.number_input("Jour Sin", -1.0, 1.0, 0.5, 0.1)
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

        X_input  = pd.DataFrame([input_data])[features]
        pred_reg = rf_reg.predict(X_input)[0]
        pred_prob= rf_clf.predict_proba(X_input)[0][1]
        pred_clf = int(pred_prob >= seuil)

        st.markdown("---")
        st.markdown("### Résultats J+3")

        col1, col2, col3 = st.columns(3)
        col1.metric("AAI Prédit J+3", f"{pred_reg:.3f}")
        col2.metric("Probabilité Dangereux", f"{pred_prob*100:.1f}%")

        if pred_clf == 1:
            col3.error("🔴 ALERTE HARMATTAN !")
            st.error("""
            ⚠️ Episode Harmattan dangereux prévu dans 3 jours !
            - Porter masques de protection
            - Limiter activités extérieures
            - Alerter établissements scolaires
            """)
        else:
            col3.success("🟢 Qualité Air Normale")
            st.success("✅ Aucun épisode dangereux prévu dans 3 jours.")

        if pred_reg < 0:
            niveau = "🟢 Faible"
        elif pred_reg < 1:
            niveau = "🟡 Modéré"
        elif pred_reg < 2:
            niveau = "🟠 Élevé"
        else:
            niveau = "🔴 Critique"

        st.info(f"Niveau de risque : {niveau}")

elif page == "Analyse Historique":
    st.title("📊 Analyse Historique AAI 2018-2025")

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
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        labels = ['Faible (AAI<0)', 'Modéré (0-1)',
                  'Élevé (1-2)', 'Critique (>2)']
        sizes  = [
            (df['AAI'] < 0).sum(),
            ((df['AAI'] >= 0) & (df['AAI'] < 1)).sum(),
            ((df['AAI'] >= 1) & (df['AAI'] < 2)).sum(),
            (df['AAI'] >= 2).sum()
        ]
        fig2 = go.Figure(go.Pie(
            labels=labels, values=sizes,
            marker=dict(colors=['green','yellow','orange','red'])
        ))
        fig2.update_layout(title='Distribution Niveaux AAI')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### Statistiques AAI")
        st.dataframe(df['AAI'].describe().round(3))

elif page == "Evaluation Modèles":
    st.title("📈 Evaluation Complète des Modèles")

    st.markdown("### Comparaison 6 Modèles")
    df_comp = pd.DataFrame({
        'Modèle'  : ['Baseline', 'RF', 'XGBoost', 'LightGBM', 'LSTM', 'CNN-LSTM'],
        'RMSE'    : [1.216, 0.678, 0.694, 0.691, 0.677, 0.699],
        'R2'      : [-1.309, 0.281, 0.249, 0.255, 0.274, 0.227],
        'AUC-ROC' : ['N/A', '0.799', '0.797', '0.776', 'N/A', 'N/A'],
        'Recall'  : ['0.000', '0.717', '0.360', '0.380', 'N/A', 'N/A']
    })
    st.dataframe(df_comp, use_container_width=True)

    st.markdown("---")
    st.markdown("### Matrice de Confusion - RF+SMOTE+Seuil 0.45")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_clf_test_clean, y_pred_clf)

    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=['Normal','Dangereux'],
        y=['Normal','Dangereux'],
        colorscale='Blues',
        text=cm, texttemplate="%{text}",
        showscale=True
    ))
    fig_cm.update_layout(
        title='Matrice de Confusion',
        xaxis_title='Prédit',
        yaxis_title='Réel',
        height=400
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("### Courbe ROC")

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_clf_test_clean, y_proba_clf)
    auc_score   = roc_auc_score(y_clf_test_clean, y_proba_clf)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'RF+SMOTE AUC={auc_score:.3f}',
        line=dict(color='steelblue', width=2)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        name='Baseline AUC=0.500',
        line=dict(color='red', dash='dash')
    ))
    fig_roc.update_layout(
        title='Courbe ROC',
        xaxis_title='Taux Faux Positifs',
        yaxis_title='Taux Vrais Positifs',
        height=400
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Cross Validation Temporelle 5 Folds")
    df_cv = pd.DataFrame({
        'Fold'     : ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Moyenne'],
        'R2'       : [0.157, 0.176, -0.449, 0.275, 0.224, 0.076],
        'RMSE'     : [0.639, 0.666, 1.177, 0.703, 0.765, 0.790],
        'AUC-ROC'  : [0.832, 0.756, 0.555, 0.790, 0.810, 0.749]
    })
    st.dataframe(df_cv, use_container_width=True)
    st.info("""
    La variabilité entre folds (R2 std=0.266) confirme que
    le modèle est sensible à la période choisie.
    Fold 3 catastrophique = année atypique non apprise.
    Cela confirme le besoin de plus de données.
    """)

    st.markdown("---")
    st.markdown("### Résidus par Niveau AAI")
    df_res = pd.DataFrame({
        'Niveau AAI'    : ['Faible (<0)', 'Modéré (0-1)',
                           'Élevé (1-2)', 'Critique (>2)'],
        'Nb observations': [183, 269, 85, 7],
        'Erreur moyenne' : [0.724, 0.310, 0.573, 1.708],
        'Erreur std'     : [0.523, 0.272, 0.318, 0.506]
    })
    st.dataframe(df_res, use_container_width=True)
    st.warning("""
    Erreur très élevée sur les jours critiques (AAI>2) : 1.708
    Seulement 7 observations de test !
    Le modèle n a pas appris suffisamment ces situations extrêmes.
    """)

elif page == "Interprétabilité":
    st.title("🔍 Interprétabilité du Modèle")

    st.markdown("### Variables les plus importantes (SHAP)")
    shap_data = {
        'Feature'    : ['AAI_roll14', 'AAI_roll30', 'AAI_lag2',
                        'AAI_lag30', 'AAI_lag3', 'AAI_lag7',
                        'LST_lag7', 'AAI_lag14', 'LST', 'pression'],
        'Importance' : [0.345, 0.194, 0.084, 0.044, 0.040,
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
        title='Top 10 Variables - Régression AAI J+3',
        xaxis_title='Importance SHAP',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interprétation physique")
    st.markdown("""
    | Variable | Importance | Signification |
    |---|---|---|
    | AAI_roll14 | 34.5% | Tendance Harmattan 14 derniers jours |
    | AAI_roll30 | 19.4% | Tendance long terme 30 jours |
    | AAI_lag2 | 8.4% | Signal avant-hier |
    | LST_lag7 | 2.7% | Chaleur semaine passée |
    | pression | 1.8% | Signal météorologique |

    > Les moyennes glissantes dominent : le Harmattan est persistant !
    """)

    st.markdown("### Métriques finales")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Régression AAI J+3**")
        st.metric("RMSE", "0.678")
        st.metric("R²", "0.281")
        st.metric("R² Cross Validation", "0.076 ± 0.266")

    with col2:
        st.markdown("**Classification Harmattan J+3**")
        st.metric("AUC-ROC", "0.799")
        st.metric("Recall", "71.7%")
        st.metric("AUC Cross Validation", "0.749 ± 0.100")

st.sidebar.markdown("---")
st.sidebar.markdown("**NABDA Issaka Joel**")
st.sidebar.markdown("Formation Data Scientist")
st.sidebar.markdown("Ouagadougou, Burkina Faso")