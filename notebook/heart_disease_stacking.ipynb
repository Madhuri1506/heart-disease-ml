# ❤️ Heart Disease Prediction using Stacking Ensemble

#imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier

#load data

df = pd.read_csv('../data/heart.csv')
df.head()

#EDA

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.countplot(x='target', data=df)
plt.title("Target Distribution")
plt.show()

#preprocessing

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LogisticRegression()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()

estimators = [
    ('lr', lr),
    ('rf', rf),
    ('gb', gb)
]

stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)


models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "Stacking": stack_model
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    results[name] = [acc, roc]

results_df = pd.DataFrame(results, index=["Accuracy", "ROC-AUC"]).T
results_df

results_df.plot(kind='bar', figsize=(8,5))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()

import pickle

pickle.dump(stack_model, open('../models/model.pkl', 'wb'))


