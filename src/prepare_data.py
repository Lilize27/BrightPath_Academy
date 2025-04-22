import pandas as pd

df = pd.read_csv("_data/Student_performance_data.csv")

print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

#Note to self, add more diagrams to analise each part!

# Univariate Analysis
sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(x='ParentalEducation', data=df)
plt.title("Parental Education count")
plt.xticks([0, 1, 2, 3, 4], ['None', 'High School','Some College','Bachelors', 'Higher Study'])
plt.tight_layout()
plt.savefig("PE_distribution.png")
#plt.show()
plt.close()

plt.figure(figsize=(4, 3))
sns.countplot(x='Tutoring', data=df, palette=["#FFAD60", "#FF6F00"])
plt.title("Tutoring Participation")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Tutoring")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("tutoring_univariate.png")
plt.show()

# plt.figure(figsize=(6, 4))
# sns.histplot(df['Age'], bins=5, kde=True)
# plt.title("Distribution of Student Age")
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.savefig("age_distribution.png")  

# plt.figure(figsize=(6, 4))
# sns.countplot(x='Gender', data=df)
# plt.title("Gender Count")
# plt.xticks([0, 1], ['Male', 'Female'])
# plt.tight_layout()
# plt.savefig("gender_distribution.png")
# #plt.show()
# plt.close()

# plt.figure(figsize=(6, 4))
# sns.countplot(x='Ethnicity', data=df)
# plt.title("Ethnicity count")
# plt.xticks([0, 1, 2, 3], ['Caucasian', 'African American','Asian','Other'])
# plt.tight_layout()
# plt.savefig("ethnicity_distribution.png")
# #plt.show()
# plt.close()



# # Bivariate Analysis
# plt.figure(figsize=(8, 5))
# sns.boxplot(x='GradeClass', y='GPA', data=df)
# plt.title("GPA Distribution by Grade Class")
# plt.tight_layout()
# plt.savefig("gpa_by_gradeclass.png")
# plt.close()

orange_palette = sns.color_palette("Oranges", n_colors=df['GradeClass'].nunique())

# plt.figure(figsize=(6, 4))
# sns.countplot(x='Gender', hue='GradeClass', data=df, palette=orange_palette)
# plt.title("Gender vs GradeClass")
# plt.xticks([0, 1], ['Male', 'Female'])
# plt.tight_layout()
# plt.savefig("gender_vs_gradeclass.png")
# plt.close()

# df['StudyTimeBin'] = pd.cut(df['StudyTimeWeekly'], bins=5)

# plt.figure(figsize=(6, 4))
# sns.barplot(x='StudyTimeBin', y='GPA', data=df, palette="Oranges")
# plt.title("Average GPA by Study Time Bins")
# plt.xlabel("Study Time Weekly (Binned)")
# plt.ylabel("Average GPA")
# plt.xticks(rotation=30)
# plt.tight_layout()
# plt.savefig("binned_studytime_vs_gpa.png")
# plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Extracurricular', hue='GradeClass', data=df, palette='Oranges')
plt.title("Extracurricular vs GradeClass")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Extracurricular Participation")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("extracurricular_vs_gradeclass.png")
plt.close()

# Sports vs GradeClass
plt.figure(figsize=(6, 4))
sns.countplot(x='Sports', hue='GradeClass', data=df, palette='Oranges')
plt.title("Sports vs GradeClass")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Sports Participation")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sports_vs_gradeclass.png")
plt.close()

# Music vs GradeClass
plt.figure(figsize=(6, 4))
sns.countplot(x='Music', hue='GradeClass', data=df, palette='Oranges')
plt.title("Music vs GradeClass")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Music Participation")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("music_vs_gradeclass.png")
plt.close()

# plt.figure(figsize=(6, 4))
# sns.scatterplot(x='StudyTimeWeekly', y='GPA', data=df)
# plt.title("Study Time vs GPA")
# plt.tight_layout()
# plt.savefig("studytime_vs_gpa.png")
# plt.close()

# count_df = df.groupby(['ParentalEducation', 'ParentalSupport']).size().unstack()

# count_df.plot(kind='bar', stacked=True, colormap='Set2')
# plt.title("Parental Support Levels Across Education Levels")
# plt.xlabel("Parental Education Level")
# plt.ylabel("Number of Students")
# plt.legend(title="Parental Support", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig("ParentSupport_vs_Education.png")
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.boxplot(x='Sports', y='GPA', data=df, palette='Set2')
# plt.title('GPA Distribution by Sports Participation')
# plt.xlabel('Sports Participation (0 = No, 1 = Yes)')
# plt.ylabel('GPA')
# plt.savefig("GPA-Distribution_vs_Sport.png")
# plt.show()

# sns.stripplot(x='Sports', y='GPA', data=df, jitter=True, palette='Set1')
# plt.title('Individual GPA Points by Sports Participation')
# plt.xlabel('Sports Participation (0 = No, 1 = Yes)')
# plt.ylabel('GPA')

# plt.show()

# plt.figure(figsize=(6, 4))
# sns.regplot(x='StudyTimeWeekly', y='GPA', data=df, scatter_kws={"color": "orange"}, line_kws={"color": "darkred"})
# plt.title("Regression Line: Study Time vs GPA")
# plt.xlabel("Study Time Weekly (hrs)")
# plt.ylabel("GPA")
# plt.tight_layout()
# plt.savefig("studytime_vs_gpa_regression.png")
# plt.show()

