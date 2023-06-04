# Importing necessary libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Display the first few rows of the data
print(data.head())

# Basic information about the data
print(data.info())

# Checking for missing values
print(data.isnull().sum())

# Correlation matrix
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of housing prices
sns.histplot(data['PRICE'], kde=True)
plt.title('Distribution of Housing Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Relationship between average number of rooms and house price
sns.scatterplot(x='RM', y='PRICE', data=data)
plt.title('Average Number of Rooms vs House Price')
plt.xlabel('Average Number of Rooms')
plt.ylabel('House Price')
plt.show()
