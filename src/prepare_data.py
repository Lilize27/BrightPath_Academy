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
#plt.show()
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x='Ethnicity', data=df)
plt.title("Ethnicity count")
plt.xticks([0, 1, 2, 3], ['Caucasian', 'African American','Asian','Other'])
plt.tight_layout()
plt.savefig("ethnicity_distribution.png")
#plt.show()
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x='ParentalEducation', data=df)
plt.title("Parental Education count")
plt.xticks([0, 1, 2, 3, 4], ['None', 'High School','Some College','Bacholers', 'Higher Study'])
plt.tight_layout()
plt.savefig("PE_distribution.png")
#plt.show()
plt.close()


# Bivariate Analysis
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

count_df = df.groupby(['ParentalEducation', 'ParentalSupport']).size().unstack()


count_df.plot(kind='bar', stacked=True, colormap='Set2')
plt.title("Parental Support Levels Across Education Levels")
plt.xlabel("Parental Education Level")
plt.ylabel("Number of Students")
plt.legend(title="Parental Support", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='Sports', y='GPA', data=df, palette='Set2')
plt.title('GPA Distribution by Sports Participation')
plt.xlabel('Sports Participation (0 = No, 1 = Yes)')
plt.ylabel('GPA')
plt.show()

sns.stripplot(x='Sports', y='GPA', data=df, jitter=True, palette='Set1')
plt.title('Individual GPA Points by Sports Participation')
plt.xlabel('Sports Participation (0 = No, 1 = Yes)')
plt.ylabel('GPA')
plt.show()
