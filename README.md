# 🌍 Prédiction du Stress Environnemental à Ouagadougou
## Système d Alerte Précoce par Télédétection Satellitaire

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)]()
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-API-green)]()

## 🎯 Objectif
Prédire les épisodes de Harmattan dangereux **3 jours à l avance**
à Ouagadougou (Burkina Faso) par fusion de données satellitaires.

## 🚀 Dashboard en ligne
👉 [Voir le Dashboard Streamlit](https://ouaga-stress-env-gcob34wn6leg7fcugavcnh.streamlit.app)

## 📡 Sources de Données
| Source | Variable |
|---|---|
| Sentinel-5P (ESA) | AAI Harmattan |
| MODIS | Température Sol |
| ERA5 (ECMWF) | Météo complète |

## 🤖 Résultats ML
| Tâche | Modèle | Score |
|---|---|---|
| Régression AAI J+3 | Random Forest | RMSE=0.678 R²=0.281 |
| Classification J+3 | RF+SMOTE+Seuil | AUC=0.799 Recall=71.7% |

## 📁 Structure
```
ouaga-stress-env/
├── app.py
├── requirements.txt
├── rapport_synthetique.docx
├── data/
├── models/
├── notebooks/
└── outputs/
```

## 👤 Auteur
Formation Data Scientist - Projet Smart City
