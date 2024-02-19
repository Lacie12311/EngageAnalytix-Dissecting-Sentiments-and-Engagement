#!/usr/bin/env python
# coding: utf-8

# + Lacie Leone
# + CIS480
# + 1/31/24

# In[8]:


import pandas as pd

# Load the dataset
file_path = 'sentimentdataset.csv'
data = pd.read_csv(file_path)

# Basic Overview of the Dataset
print("Dataset Overview:")
print(data.describe(include='all'))
print("\n")

# Count of Unique Values in Each Column
print("Unique Values in Each Column:")
for col in data.columns:
    print(f"{col}: {data[col].nunique()} unique values")
print("\n")


# In[10]:


import matplotlib.pyplot as plt

# Distribution of Sentiments
print("Distribution of Sentiments:")
sentiment_counts = data['Sentiment'].value_counts()
print(sentiment_counts)

# Print the plot
plt.figure(figsize=(10,8))  # Increase figure size
sentiment_counts[:20].plot(kind='bar', title='Sentiment Distribution')  # Show top 20 sentiments only
plt.xticks(rotation=45, ha='right')  # Rotate x-tick labels for better readability
plt.tight_layout()  # Adjust layout
plt.show()


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the Data
data = pd.read_csv('sentimentdataset.csv')

# Summary Statistics for Numerical Columns
print("Summary Statistics for Numerical Columns:")
summary_statistics = data.describe()
print(summary_statistics)


# In[14]:


# Correlation Analysis
# We have numerical columns like 'Likes' and 'Retweets' to correlate.
# Replace 'Likes' and 'Retweets' with the actual numerical columns.
print("\nCorrelation Matrix:")
correlation_matrix = data[['Likes', 'Retweets']].corr() 
print(correlation_matrix)


# In[15]:


# Categorical Data Analysis
# This shows 'Platform' is a categorical column in the dataset.
# Replace 'Platform' with the actual categorical column to analyze.
print("\nPosts by Platform:")
posts_by_platform = data['Platform'].value_counts()
print(posts_by_platform)

# Plot the posts by platform as a bar chart for better visualization.
plt.figure(figsize=(10, 6))
posts_by_platform.plot(kind='bar')
plt.title('Number of Posts by Platform')
plt.xlabel('Platform')
plt.ylabel('Number of Posts')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:




