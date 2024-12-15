

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#ensuure vgsales.csv is in the same directory as this py file
import pandas as pd
filepath = "/vgsales.csv"
df = pd.read_csv(filepath)
df.head()

#PREPROCESSING

# For proper scaling, I will also convert the Global_Sales (in millions) into normal values 1.2 million -> 1,200,000
df['Global_Sales'] = df['Global_Sales'] * 1000000
df = df.dropna(subset=['Year', 'Publisher'])
#drop rows with missing values, they make up a very small portion of dataset so it's ok to drop
df.info()

min= int(df['Year'].min())
max = int(df['Year'].max())
print(f"The years of games range from {min} to {max}")

unique_platforms = df['Platform'].nunique()
unique_publishers = df['Publisher'].nunique()
unique_genres = df['Genre'].nunique()
unique_names = df['Name'].nunique()

print(f"Number of unique platforms: {unique_platforms}")
print(f"Number of unique publishers: {unique_publishers}")
print(f"Number of unique genres: {unique_genres}")
print(f"Number of unique names: {unique_names}")

#EDA

#Total global game sales is the metric we are using here
mean = df['Global_Sales'].mean()
print(f"The mean global sales is {mean} sales")

median = df['Global_Sales'].median()
print(f"The median global sales is {median} sales")

std = df['Global_Sales'].std()
print(f"The standard deviation of global sales is {std} sales")

min = df['Global_Sales'].min()
print(f"The minimum global sales is {min} sales")

max = df['Global_Sales'].max()
print(f"The maximum global sales is {max} sales")

#Creating a copy of the data for visualization
df_copy = df.copy()

# Adding a log-transformed version
df_copy['Log_Global_Sales'] = np.log(df['Global_Sales'])

# SIDE BY SIDE BOX PLOTS
plt.figure(figsize=(12, 6))

# OG Global Sales boxplot
plt.subplot(1, 2, 1)
sns.boxplot(data=df_copy, x='Global_Sales')
plt.title('Boxplot of Global Sales')

# Log-transformed Global Sales boxplot
plt.subplot(1, 2, 2)
sns.boxplot(data=df_copy, x='Log_Global_Sales')
plt.title('Boxplot of Log-Transformed Global Sales')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# OG version histogram
plt.subplot(1, 2, 1)
sns.histplot(data=df_copy, x='Global_Sales', bins=30, kde=True)
plt.title('Distribution of Global Sales')

# Log-transformed histogram of Global Sales
plt.subplot(1, 2, 2)
sns.histplot(data=df_copy, x='Log_Global_Sales', bins=30, kde=True)
plt.title('Distribution of Log-Transformed Global Sales')

plt.tight_layout()
plt.show()

# Scatter plot of Release Year vs. Global Sales
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Year', y='Global_Sales')
plt.title('Scatter Plot of Release Year vs. Global Sales')
plt.xlabel('Release Year')
plt.ylabel('Global Sales')
plt.show()
from scipy.stats import linregress

# Calculate correlation and R^2
slope, intercept, r_value, p_value, std_err = linregress(df['Year'], df['Global_Sales'])

# Print results
print(f"R (Correlation Coefficient): {r_value:.4f}")
print(f"R^2 (Coefficient of Determination): {r_value**2:.4f}")

# Correct the data preparation for bar plots
avg_sales_by_genre = df.groupby('Genre')['Global_Sales'].mean()
total_sales_by_genre = df.groupby('Genre')['Global_Sales'].sum()
avg_sales_by_genre_df = avg_sales_by_genre.reset_index().sort_values(by='Global_Sales', ascending=False)
total_sales_by_genre_df = total_sales_by_genre.reset_index().sort_values(by='Global_Sales', ascending=False)

plt.figure(figsize=(16, 6))

# AVG sales with error bars
plt.subplot(1, 2, 1)
sns.barplot(data=avg_sales_by_genre_df, x='Genre', y='Global_Sales', palette="viridis", capsize=0.1)
plt.title('Average Global Sales by Genre')
plt.xlabel('Genre')
plt.ylabel('Average Sales')
plt.xticks(rotation=45)

# TOTAL sales
plt.subplot(1, 2, 2)
sns.barplot(data=total_sales_by_genre_df, x='Genre', y='Global_Sales', palette="viridis", capsize=0.1)
plt.title('Total Global Sales by Genre')
plt.xlabel('Genre')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

avg_sales_by_region = {
    "NA_Sales": df["NA_Sales"].mean()*1000000,
    "EU_Sales": df["EU_Sales"].mean()*1000000,
    "JP_Sales": df["JP_Sales"].mean()*1000000,
    "Other_Sales": df["Other_Sales"].mean()*1000000,
    "Global_Sales": df["Global_Sales"].mean()
}
total_sales_by_region = {
    "NA_Sales": df["NA_Sales"].sum()*1000000,
    "EU_Sales": df["EU_Sales"].sum()*1000000,
    "JP_Sales": df["JP_Sales"].sum()*1000000,
    "Other_Sales": df["Other_Sales"].sum()*1000000,
    "Global_Sales": df["Global_Sales"].sum()
}


avg_sales_by_region_df = pd.DataFrame(list(avg_sales_by_region.items()), columns=["Region", "Average Sales"])
total_sales_by_region_df = pd.DataFrame(list(total_sales_by_region.items()), columns=["Region", "Total Sales"])

# Display the result
print(avg_sales_by_region_df)
print()
print(total_sales_by_region_df)

# Calculate TOTAL sales by PUBLISHER
total_sales_by_publisher = df.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=total_sales_by_publisher.values, y=total_sales_by_publisher.index, palette="gray")
plt.title('Total Global Sales by Publisher (Top 20)')
plt.xlabel('Total Sales')
plt.ylabel('Publisher')
plt.show()

# CALC TOTAL SALES OF WHOLE DATASET
total_sales = df['Global_Sales'].sum()

# Calculate TOTAL SALES BY PUBLISHER, sort desc
total_sales_by_publisher = df.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False)

# CALC the TOTAL sales of the TOP 10
top_10_publishers_sales = total_sales_by_publisher.head(10).sum()

# CALC the PROPORTION of TOTAL sales by the TOP 10
proportion_top_10 = top_10_publishers_sales / total_sales

print("Total Sales of dataset: ", total_sales)
print("Total Sales of Top 10 Publishers: ", top_10_publishers_sales)
print("Proportion of Total Sales of Top 10 Publishers: ", proportion_top_10)

# Calculate TOTAL sales by PLATFORM
total_sales_by_publisher = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=total_sales_by_publisher.values, y=total_sales_by_publisher.index, palette="gray")
plt.title('Total Global Sales by Platform (Top 10)')
plt.xlabel('Total Sales')
plt.ylabel('Platform')
plt.show()

# Calculate TOTAL SALES BY PLATFORM, sort desc
total_sales_by_platform= df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)

# CALC the TOTAL sales of the TOP 10
top_10_platform_sales = total_sales_by_platform.head(10).sum()

# CALC the PROPORTION of TOTAL sales by the TOP 10
proportion_top_10 = top_10_platform_sales / total_sales

print("Total Sales of dataset: ", total_sales)
print("Total Sales of Top 10 Platforms: ", top_10_platform_sales)
print("Proportion of Total Sales of Top 10 Platforms: ", proportion_top_10)
