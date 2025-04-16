import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

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

#  Visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=sorted(y_test.unique()), 
            yticklabels=sorted(y_test.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix of GradeClass Prediction")
plt.tight_layout()
plt.show()