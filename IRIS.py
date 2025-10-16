import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 

try:
    iris_df = pd.read_csv('IRIS.csv')
except FileNotFoundError:
    print("Error: IRIS.csv not found. Please make sure the dataset file is in the same directory.")
    exit()

def visualize_data(df):
    """Creates a pair plot to visualize relationships between features."""
    print("Displaying pair plot to show feature relationships by species...")
    sns.pairplot(df, hue='species', markers=["o", "s", "D"])
    plt.suptitle("Pair Plot of Iris Dataset Features", y=1.02)
    plt.show()

visualize_data(iris_df)

X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- {name} ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

final_model = SVC()
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("--- Hyperparameter Tuning for SVM ---")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)
grid_search.fit(X_train, y_train)

print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Accuracy with best parameters: {grid_search.best_score_ * 100:.2f}%\n")
tuned_model = grid_search.best_estimator_


print("--- Final Model Evaluation (Tuned SVM) ---")
tuned_pred = tuned_model.predict(X_test)
final_accuracy = accuracy_score(y_test, tuned_pred)

print(f"Final Model Accuracy: {final_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, tuned_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tuned_pred))

model_filename = 'iris_classifier_model.joblib'
joblib.dump(tuned_model, model_filename)
print(f"\nModel saved to '{model_filename}'")

loaded_model = joblib.load(model_filename)
print(f"Model loaded from '{model_filename}'")

new_flower_data = [[5.1, 3.5, 1.4, 0.2]] 
new_prediction = loaded_model.predict(new_flower_data)

print(f"\nPrediction for {new_flower_data} using loaded model: {new_prediction[0]}")