import pandas as pd


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

 #  numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Define the IQR outlier function
def find_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers

# Run the check
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