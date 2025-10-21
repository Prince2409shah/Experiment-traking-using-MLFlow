import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import joblib
import os

# Initialize DagsHub + MLflow
dagshub.init(repo_owner='Prince_Shah', repo_name='Experiment-traking-using-MLFlow', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Prince_Shah/Experiment-traking-using-MLFlow.mlflow')

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Model parameters
max_depth = 10
n_estimators = 5

# Create / select experiment
mlflow.set_experiment('MLOPS-Exp1')

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # Log the confusion matrix image
    mlflow.log_artifact('confusion_matrix.png')

    # Log the training script if available
    if '__file__' in globals():
        mlflow.log_artifact(__file__)

    # Log model manually (to avoid unsupported endpoint)
    model_filename = 'random_forest_model.pkl'
    joblib.dump(rf, model_filename)
    mlflow.log_artifact(model_filename)

    # Set tags
    mlflow.set_tags({
        "model": "RandomForestClassifier",
        "Project": "Wine Classification"
    })

    print(f"✅ Run complete! Accuracy: {accuracy:.4f}")
