import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# CHanging the file url to https 

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5

# Mention your experiment name
mlflow.set_experiment('MLOPS-Exp1')
# Start MLflow run to log experiment
with mlflow.start_run():
    # Initialize and train the Random Forest Classifier
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters to MLflow
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Creating a confusion matrix plot
    cm=confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    #save the plot to a file
    plt.savefig('confusion_matrix.png')

    #log artifacts using mlflow
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    # tags
    mlflow.set_tags({"model":"RandomForestClassifier","Project":"Wine Classification"})
    # Log the model 
    mlflow.sklearn.log_model(rf, "random_forest_model")

    # Print the result
    print(f"Accuracy: {accuracy}")