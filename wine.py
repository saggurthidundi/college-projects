import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('WineQT.csv')

print(df.describe())

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print(corr_matrix)

highly_correlated = corr_matrix[(abs(corr_matrix) > 0.6) & (corr_matrix != 1.0)].dropna(how='all').dropna(axis=1, how='all')
print(highly_correlated)

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
outliers_count = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
print(outliers_count)

if 'quality' in df.columns and 'alcohol' in df.columns and 'fixed acidity' in df.columns:
    high_quality_wines = df[df['quality'] >= 7].sort_values(by='alcohol', ascending=False)
    print(high_quality_wines[['fixed acidity', 'alcohol', 'quality']].head())

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Wine Features')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

if 'quality' in df.columns and 'alcohol' in df.columns:
    sns.barplot(x='quality', y='alcohol', data=df, ax=axes[0], palette='viridis')
    axes[0].set_title('Alcohol Content by Wine Quality')

if 'pH' in df.columns:
    sns.histplot(df['pH'], bins=20, kde=True, ax=axes[1], color='darkred')
    axes[1].set_title('Distribution of pH Levels')

if 'quality' in df.columns and 'fixed acidity' in df.columns:
    sns.boxplot(x='quality', y='fixed acidity', data=df, ax=axes[2], palette='Set3')
    axes[2].set_title('Fixed Acidity by Quality')

plt.tight_layout()
plt.show()
