import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('diamonds.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df = df.drop_duplicates()

df[['x', 'y', 'z']] = df[['x', 'y', 'z']].replace(0, np.nan)
df = df.dropna()

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

df['price'] = np.where(df['price'] > upper_bound, upper_bound, df['price'])

print(df.describe())

print(df.describe(include=['object']))

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print(corr_matrix)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.scatterplot(x='carat', y='price', data=df, ax=axes[0, 0], alpha=0.3, color='teal')
axes[0, 0].set_title('Price vs. Carat Weight')

sns.boxplot(x='cut', y='price', data=df, order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Price by Cut Quality')

sns.boxplot(x='color', y='price', data=df, order=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ax=axes[1, 0], palette='coolwarm')
axes[1, 0].set_title('Price by Color Grade')

sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f', ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.show()
