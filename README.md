Détection d'anomalies sur télémétrie satellite — ESA OPS-SAT

Pipeline de données et de machine learning sur Databricks, en PySpark, pour détecter automatiquement les anomalies dans la télémétrie du satellite OPS-SAT de l'Agence spatiale européenne.

Résultat : F1 de 0,945 et AUC-PR de 0,991 sur le jeu de test officiel, comparable au meilleur modèle publié sur ce benchmark (FCNN : F1 0,946, AUC-PR 0,979).

Contexte

Les opérateurs satellite surveillent des milliers de signaux de télémétrie. Repérer à la main un comportement anormal est long, coûteux et sujet à l'erreur. Ce projet construit une chaîne complète, de la donnée brute jusqu'à un modèle évalué, sur le jeu public OPSSAT-AD :

303 493 mesures brutes ;
2 123 segments de télémétrie, répartis sur 9 canaux ;
environ 20 % de segments anormaux, annotés par des ingénieurs d'opérations spatiales.


Architecture médaillon sur Databricks, avec Unity Catalog et Delta Lake
```
                        ┌──────────────────────────┐
                        │  dataset.csv             │
                        │  features officielles    │
                        └────────────┬─────────────┘
                                     │ validation
                                     ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ segments.csv │──▶│    BRONZE    │──▶│    SILVER    │──▶│     GOLD     │
│ 303 493      │   │ donnée brute │   │ typage,      │   │ 29 features  │
│ mesures      │   │ + traçabilité│   │ nettoyage,   │   │ par segment  │
│              │   │              │   │ contrôles    │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
                                                                │
                                   ┌────────────────────────────┤
                                   ▼                            ▼
                         ┌──────────────────┐         ┌──────────────────┐
                         │  MODÉLISATION    │         │     SCORING      │
                         │  expériences     │         │  table de        │
                         │  MLflow          │         │  prédictions     │
                         └────────┬─────────┘         └────────┬─────────┘
                                  │                     ▲      │
                                  ▼                     │      ▼
                         ┌──────────────────┐           │ ┌──────────────────┐
                         │ REGISTRE UC      │───────────┘ │   DASHBOARD      │
                         │ opssat_anomaly_  │  @champion  │   surveillance   │
                         │ xgb              │             │   des alertes    │
                         └──────────────────┘             └──────────────────┘

        Job opssat_pipeline : bronze → silver → gold → scoring

Bronze:	bronze_segments	Télémétrie brute, colonnes en texte, date d'ingestion et fichier source
Silver:	silver_telemetry	Données typées, dédoublonnées, contrôlées
Gold:	gold_segment_features	18 features de l'article, recalculées en PySpark
Gold:	gold_features_extra	11 features supplémentaires conçues pour ce projet
Modèle:	opssat_anomaly_xgb	Modèle XGBoost versionné dans Unity Catalog, alias champion
Prédictions:	predictions_anomalies	Probabilité d'anomalie et alerte par segment, avec version du modèle et date de scoring

Les notebooks
Notebook	Rôle	Dans le Job
01_exploration	Première exploration du jeu de données	Non
02_Bronze	Ingestion brute de segments.csv en table Delta	Oui
03_Silver	Typage, nettoyage, contrôles qualité	Oui
04_Gold	Feature engineering par fonctions de fenêtre, validation contre la référence	Oui
06_Modelisation	Expériences : comparaison de modèles, optimisation, évaluation, enregistrement du modèle	Non, lancé à la main
07_Scoring	Chargement du modèle @champion, prédictions, écriture de la table d'alertes	Oui

Les expériences sont séparées du pipeline : le Job ne refait pas l'entraînement à chaque exécution, il applique le modèle validé aux données.

Étapes et résultats
1. Qualité des données (Silver)
Colonne label retirée : constante sur toutes les lignes, sans information.
2 123 segments et 9 canaux retrouvés, comme dans la référence.
0 segment incohérent : chaque segment a un seul canal, un seul label et un seul split.
2. Feature engineering et validation (Gold)

Les 18 features de l'article (statistiques, dérivées, pics, irrégularité d'échantillonnage) sont recalculées en PySpark avec des fonctions de fenêtre (lag, lead, moyennes glissantes), puis comparées segment par segment au fichier officiel :

Résultat	Features
Identiques (écart de l'ordre de 10⁻¹⁶)	14 features : durée, longueur, moments statistiques, variances, irrégularité
Différentes	4 features de pics : l'article utilise un seuil de proéminence minimale de 10 %

Une erreur de variance (variance d'échantillon au lieu de variance de population) a été détectée et corrigée grâce à cette validation.

À modèle identique, les pics officiels améliorent le F1 d'environ 0,03, de façon stable sur 5 graines. Ils sont donc utilisés pour la suite.

3. Choix du modèle

Comparaison sur le split officiel, moyenne sur 5 graines :

Modèle	F1	Recall
Régression logistique (baseline)	0,844	0,885
Random Forest	0,904	0,855
XGBoost	0,911	0,903

XGBoost est retenu pour son meilleur recall : dans un contexte d'opérations spatiales, rater une anomalie coûte plus cher qu'une fausse alerte.

4. Optimisation

Recherche de 40 combinaisons d'hyperparamètres en validation croisée à 5 parts, puis réglage du seuil d'alerte sur des prédictions out-of-fold.

Conclusion : aucun gain significatif. Le F1 reste stable entre 0,90 et 0,91 pour des seuils de 0,25 à 0,65. Le modèle est limité par l'information contenue dans ses features, pas par ses réglages.

5. Nouvelles features

11 features conçues et calculées en PySpark : minimum, maximum, amplitude, médiane, écart interquartile, saut maximal et moyen entre deux mesures, nombre de changements de sens, pente (régression linéaire), autocorrélation, plus grand trou de télémétrie.

Jeu de features	F1 (validation croisée, 5 parts × 3)	AUC-PR
18 features de l'article	0,894 ± 0,039	0,961 ± 0,017
18 + 11 nouvelles	0,913 ± 0,031	0,973 ± 0,013

Le gain est confirmé par une comparaison appariée : les nouvelles features gagnent sur 12 découpages sur 15, gain moyen de +0,019.

Le plus grand trou de télémétrie (trou_max) devient la 2ᵉ variable la plus importante du modèle, derrière le nombre de pics.

6. Évaluation finale

Le jeu de test n'a été utilisé qu'une fois, à la fin, pour éviter de régler le modèle sur ses données.

Modèle	F1	Precision	Recall	AUC-PR
FCNN, meilleur modèle de l'article	0,946	0,963	0,929	0,979
Ce projet : XGBoost, 29 features	0,945	0,981	0,912	0,991

Ce résultat repose sur un jeu de test de 113 anomalies : il est comparable au meilleur résultat publié, sans prétendre le dépasser.

Mise en production
7. Enregistrement du modèle

Le modèle final est enregistré avec MLflow dans le registre de modèles Unity Catalog (workspace.default.opssat_anomaly_xgb) :

une signature décrit les colonnes attendues en entrée : des données mal formées sont refusées au lieu de produire une prédiction fausse ;
un exemple d'entrée documente le format ;
l'alias champion désigne la version en service. Une nouvelle version se met en production en déplaçant l'alias, sans modifier le code de scoring.
8. Scoring

Le notebook 07_Scoring recalcule les features exactement comme à l'entraînement (mêmes tables, mêmes jointures, même ordre de colonnes), charge le modèle @champion et écrit la table predictions_anomalies.

Chaque prédiction porte la version du modèle et la date de scoring : une alerte reste traçable jusqu'au modèle qui l'a produite.

9. Orchestration

Le Job Databricks opssat_pipeline enchaîne quatre tâches sur compute Serverless :

bronze → silver → gold → scoring

Chaque tâche ne démarre que si la précédente a réussi : un échec en Silver empêche le calcul du Gold sur des données incomplètes.

<!-- Capture d'écran du graphe des tâches du Job -->
10. Dashboard de surveillance

Un dashboard Databricks construit sur predictions_anomalies, pensé pour un opérateur :

Vue	Usage
Indicateurs clés	Nombre de segments, d'alertes, part d'alertes, version du modèle, date du dernier scoring
Alertes par canal	Repérer les capteurs les plus perturbés
Performance par canal (jeu de test)	Bonnes détections, anomalies ratées et fausses alertes par capteur
Alertes à examiner	Liste de travail triée par probabilité décroissante
Cas incertains	Segments dont la probabilité est entre 0,2 et 0,8, à vérifier en priorité

Les cas incertains sont aussi les plus utiles pour améliorer le modèle : une fois vérifiés par un opérateur, ils fournissent les exemples difficiles qui lui manquent.


Ce que montre ce projet
Data engineering : pipeline médaillon PySpark, tables Delta, contrôles qualité, validation contre une référence, orchestration par Job.
Feature engineering : fonctions de fenêtre, agrégations, régression et corrélation calculées dans Spark.
Évaluation rigoureuse : baseline, validation croisée, comparaisons appariées, test utilisé une seule fois.
MLOps : suivi des expériences dans MLflow, modèle versionné dans Unity Catalog avec signature et alias, scoring traçable.
Restitution : dashboard de surveillance orienté opérateur.

Lancer le projet
Créer un compte Databricks (la Free Edition suffit).
Télécharger segments.csv et dataset.csv depuis Zenodo.
Les déposer dans un volume Unity Catalog, puis ajuster CATALOG, SCHEMA et VOLUME dans la première cellule de chaque notebook.
Lancer 02_Bronze, 03_Silver et 04_Gold.
Lancer 06_Modelisation pour entraîner et enregistrer le modèle, puis lui donner l'alias champion.
Lancer 07_Scoring, ou le Job opssat_pipeline pour enchaîner toute la chaîne.

Les notebooks de modélisation et de scoring installent xgboost dans leur première cellule.

Pistes d'amélioration
Analyser les erreurs du modèle (anomalies ratées, fausses alertes) par canal.
Normaliser les features par canal.
Ajouter des features fréquentielles (transformée de Fourier).
Recalculer les pics avec un seuil de proéminence pour un pipeline entièrement autonome.
Étendre le pipeline à l'ESA Anomaly Dataset (3 missions, environ 31 Go).

Sources
Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. The OPS-SAT benchmark for detecting anomalies in satellite telemetry, Scientific Data, 2025. arXiv:2407.04730
Jeu de données OPSSAT-AD : Zenodo
