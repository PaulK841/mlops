import os
import mlflow
import mlflow.sklearn
import dagshub
import joblib # Librairie pour sauvegarder le modèle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# =================================================================
# 1. CONFIGURATION POUR DAGSHUB
# =================================================================
# On initialise la connexion à VOTRE repo.
# mlflow=True configure automatiquement MLflow pour logger sur DagsHub.
dagshub.init(repo_owner='paulker194', repo_name='mlops', mlflow=True)

# =================================================================
# 2. CHARGEMENT DES DONNÉES ET PRÉPARATION
# =================================================================
print("Loading 'digits' dataset from scikit-learn...")
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# On définit le nom de l'expérience sur DagsHub
mlflow.set_experiment("Digits-Classification-Experiment")

# =================================================================
# 3. FONCTION D'ENTRAÎNEMENT ET LOGGING
# =================================================================
def train_and_log_model(n_estimators, max_depth):
    # Démarrer une nouvelle exécution (run) MLflow
    with mlflow.start_run() as run:
        print(f"--- Training with n_estimators={n_estimators}, max_depth={max_depth} ---")
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Entraîner le modèle
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Faire les prédictions et calculer l'accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")

        # Log les hyperparamètres sur DagsHub
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log la métrique sur DagsHub
        mlflow.log_metric("accuracy", accuracy)

        # --- CORRECTION APPLIQUÉE ICI ---
        # Sauvegarder le modèle dans un fichier local temporaire
        model_filename = "model.joblib"
        joblib.dump(model, model_filename)

        # Logger ce fichier comme un artefact.
        # Le deuxième argument est le dossier où il apparaîtra dans l'interface MLflow.
        mlflow.log_artifact(model_filename, artifact_path="model")
        
        # Supprimer le fichier local après l'avoir loggé
        os.remove(model_filename)
        
        print("Model logged as an artifact to DagsHub successfully.")

# =================================================================
# 4. LANCER LES EXPÉRIENCES
# =================================================================
# Lancer la première expérience
train_and_log_model(n_estimators=20, max_depth=5)

# Lancer la seconde expérience
train_and_log_model(n_estimators=100, max_depth=10)

print("\nAll experiments are complete!")
print("Go to https://dagshub.com/paulker194/mlops/experiments/ to see your results.")