from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

# Load dataset
iris = load_iris()

# Create DataFrame
X = pd.DataFrame(iris.data, columns=[
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
])
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Save model
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully")