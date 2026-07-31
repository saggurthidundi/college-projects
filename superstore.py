import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Sample - Superstore.csv', encoding='windows-1252')

df = df.drop_duplicates()
df = df.dropna()

df['Order Date'] = pd.to_datetime(df['Order Date'])

df = df.sort_values(by='Order Date')

high_value_orders = df[df['Sales'] > 1000]

sales_profit_stats = df[['Sales', 'Profit']].describe()
print(sales_profit_stats)

category_perf = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
print(category_perf)

region_perf = df.groupby('Region')[['Sales', 'Profit']].sum().reset_index()
print(region_perf)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.barplot(x='Category', y='Sales', data=category_perf, ax=axes[0, 0], palette='pastel')
axes[0, 0].set_title('Total Sales by Category')

monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()
sns.lineplot(x='Order Date', y='Sales', data=monthly_sales, ax=axes[0, 1], color='coral', marker='o')
axes[0, 1].set_title('Monthly Sales Trend')
axes[0, 1].tick_params(axis='x', rotation=45)

sns.histplot(df['Profit'], bins=100, kde=True, ax=axes[1, 0], color='purple')
axes[1, 0].set_xlim(-500, 500) 
axes[1, 0].set_title('Distribution of Profit')

sns.scatterplot(x='Sales', y='Profit', hue='Category', data=df, alpha=0.6, ax=axes[1, 1])
axes[1, 1].set_title('Sales vs. Profit by Category')

plt.tight_layout()
plt.show()
