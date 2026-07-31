import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')

df = df.drop_duplicates()

age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)

embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

if 'Cabin' in df.columns:
    df = df.drop(columns=['Cabin'])

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

df['Fare'] = np.where(df['Fare'] > upper_bound, upper_bound, df['Fare'])

filtered_df = df[(df['Age'] > 18) & (df['Sex'] == 'female') & (df['Pclass'].isin([1, 2]))]
print(filtered_df[['Name', 'Age', 'Pclass']].head())

survival_stats = df.groupby(['Pclass', 'Sex'])['Survived'].mean().reset_index()
print(survival_stats)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x='Pclass', y='Survived', data=df, ax=axes[0], palette='Blues_d')
axes[0].set_title('Survival Rate by Passenger Class')
axes[0].set_ylabel('Survival Probability')

sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=20, ax=axes[1])
axes[1].set_title('Age Distribution Stacked by Survival')

sns.barplot(x='Sex', y='Survived', data=df, ax=axes[2], palette='Set2')
axes[2].set_title('Survival Rate by Sex')
axes[2].set_ylabel('Survival Probability')

plt.tight_layout()
plt.show()
