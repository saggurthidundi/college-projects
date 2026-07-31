import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
# Ensure 'StudentsPerformance.csv' is uploaded to your Colab environment
df = pd.read_csv('StudentsPerformance.csv')

# 2. Exploratory Data Analysis & 3. Handle missing values and duplicates
print("--- Dataset Info ---")
print(df.info())
df = df.drop_duplicates()

# Fill missing numeric values with median, categorical with mode (if any exist)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# 4. Calculate mean, median, mode, variance, and standard deviation
numeric_cols = df.select_dtypes(include=[np.number]).columns
stats_df = pd.DataFrame({
    'Mean': df[numeric_cols].mean(),
    'Median': df[numeric_cols].median(),
    'Variance': df[numeric_cols].var(),
    'Std_Dev': df[numeric_cols].std()
})
print("\n--- Statistical Measures ---")
print(stats_df)
print("\nMode of all columns:\n", df.mode().iloc[0])

# 5. Detect and treat outliers (Using Interquartile Range on 'math score')
Q1 = df['math score'].quantile(0.25)
Q3 = df['math score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Treat outliers by capping them to the bounds
df['math score'] = np.where(df['math score'] < lower_bound, lower_bound, df['math score'])
df['math score'] = np.where(df['math score'] > upper_bound, upper_bound, df['math score'])

# 6. Analyze correlations between variables
print("\n--- Correlation Matrix ---")
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)

# 7. Create Visualizations
plt.figure(figsize=(15, 10))

# Bar Chart
plt.subplot(2, 2, 1)
sns.barplot(x='gender', y='math score', data=df, palette='muted')
plt.title('Average Math Score by Gender')

# Line Chart (Tracking first 50 students' reading scores)
plt.subplot(2, 2, 2)
plt.plot(df['reading score'].head(50), marker='o', color='green')
plt.title('Reading Score Trend (First 50 Students)')
plt.xlabel('Student Index')
plt.ylabel('Reading Score')

# Histogram
plt.subplot(2, 2, 3)
sns.histplot(df['writing score'], bins=15, kde=True, color='purple')
plt.title('Distribution of Writing Scores')

# Scatter Plot
plt.subplot(2, 2, 4)
sns.scatterplot(x='math score', y='reading score', hue='gender', data=df, alpha=0.7)
plt.title('Math vs Reading Score')

plt.tight_layout()
plt.show()
