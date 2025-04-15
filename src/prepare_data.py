import pandas as pd


df = pd.read_csv("_data/Student_performance_data.csv")


print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

#Note to self, add more diagrams to analise each part!

# Univariate Analysis
sns.set_theme(style="whitegrid")


plt.figure(figsize=(6, 4))
sns.histplot(df['Age'], bins=5, kde=True)
plt.title("Distribution of Student Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("age_distribution.png")  


plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=df)
plt.title("Gender Count")
plt.xticks([0, 1], ['Male', 'Female'])
plt.tight_layout()
plt.savefig("gender_distribution.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x='GradeClass', y='GPA', data=df)
plt.title("GPA Distribution by Grade Class")
plt.tight_layout()
plt.savefig("gpa_by_gradeclass.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', hue='GradeClass', data=df)
plt.title("Gender vs GradeClass")
plt.xticks([0, 1], ['Male', 'Female'])
plt.tight_layout()
plt.savefig("gender_vs_gradeclass.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='StudyTimeWeekly', y='GPA', data=df)
plt.title("Study Time vs GPA")
plt.tight_layout()
plt.savefig("studytime_vs_gpa.png")
plt.close()