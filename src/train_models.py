import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

#Model 1

#Predicting the GradeClass using Random Forest Classifier
#Training Time
train_df = pd.read_csv("_data/cleaned_train_data.csv")
test_df = pd.read_csv("_data/cleaned_test_data.csv")

X_train = train_df.drop("GradeClass", axis=1)
y_train = train_df["GradeClass"]
#Drop from categories
X_test = test_df.drop("GradeClass", axis=1)
y_test = test_df["GradeClass"]

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation report
print(" Classification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

#  Visualization of the confusion matrix, and let me tell you I am confused
plt.figure(figsize=(8, 6))

grade_labels = ['A', 'B', 'C', 'D', 'F']
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of GradeClass Prediction")
plt.tight_layout()
plt.savefig("RandomForest_Confusion-matrix_Heat.png")
plt.show()

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model.estimators_[0], feature_names=X_train.columns, class_names=grade_labels, filled=True, max_depth=3)
plt.title("One Tree from the Random Forest (Depth=3)")
plt.savefig("OneTree-Depth3.png")
plt.show()

#Model 2
#Predicting the GradeClass using XGBoost Classifier

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

# Evaluation
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

importances = model.feature_importances_
feature_names = X_train.columns

# Plot
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
feat_imp.plot(kind='bar', figsize=(10,6), title='Feature Importances')
plt.tight_layout()
plt.savefig("FeautureImportance.png")
plt.show()

import joblib  # or use pickle

# Assuming these are your models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Save RandomForest model
joblib.dump(model, 'artifacts/random_forest_model.pkl')

# Save XGBoost model
joblib.dump(xgb_model, 'artifacts/xgb_model.pkl')
