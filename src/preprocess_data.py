import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("_data/Student_performance_data.csv")


print(df.head())

#print(df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt


#sns.boxplot(x=df['Absences'])


#plt.title("Boxplot of Absences")
#plt.show()

#Q1 = df['Absences'].quantile(0.25)
#Q3 = df['Absences'].quantile(0.75)
#IQR = Q3 - Q1

#lower_bound = Q1 - 1.5 * IQR
#upper_bound = Q3 + 1.5 * IQR

# Detect outliers
#outliers = df[(df['Absences'] < lower_bound) | (df['Absences'] > upper_bound)]
#print(outliers)

#seems like there is no outliers or missing values...I dont trust it though

print("Missing values per column:")
print(df.isnull().sum())

#double check boxplot
if df.isnull().values.any():
    print("There are missing values in the dataset.")
else:
    print("No missing values found!")

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

 

def find_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers


print("\n Checking for outliers:")
for col in numeric_cols:
    outliers = find_outliers_iqr(df, col)
    if not outliers.empty:
        print(f" {len(outliers)} outlier(s) found in '{col}'")
    else:
        print(f" No outliers in '{col}'")


#exceptable values are lower than 4 and 1
print("ParentalEducation:", df['ParentalEducation'].unique())
print("Music:", df['Music'].unique())
print("Volunteering:", df['Volunteering'].unique())


#Verdict is to keep outliers as there are no unique or error values.

print(df.info())
print(df.isnull().sum())  # check for missing values
print(df.duplicated().sum())

df.dropna(inplace=True)  # Drop rows with missing values

df.drop(columns=['StudentID'], inplace=True)

df['GradeClass'] = pd.to_numeric(df['GradeClass'], errors='coerce')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['StudyTimeWeekly', 'Absences', 'GPA']] = scaler.fit_transform(df[['StudyTimeWeekly', 'Absences', 'GPA']]) #rescales for better performance

X = df.drop('GradeClass', axis=1)
y = df['GradeClass']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)  # keeps class balance

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("cleaned_train_data.csv", index=False)
test_df.to_csv("cleaned_test_data.csv", index=False)