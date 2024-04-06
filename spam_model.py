"""
NLP project to classify spam messages using Naive Bayes Classifier
Group 6 - W24
Members: Helia Mozaffari, Susmita Roy, Sandeep Neupane, Diego Sarmiento
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud

# Load the data

path = './COMP237_NPL_Project/src/Data6.csv'
input_data = pd.read_csv(path, encoding='latin-1')

# Data exploration
"""
print(input_data.info())
print(input_data.head())
print(input_data.shape)

"""
label_counts = input_data['Label'].value_counts()
print(label_counts)

plt.figure(figsize=(6, 5))
plt.bar(label_counts.index, label_counts.values, color=['green', 'red'])
plt.title('Label distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(label_counts.index, ['No spam', 'Spam'], rotation=0)
#plt.show()


spam = input_data[input_data['Label'] == 1]['Body']
no_spam = input_data[input_data['Label'] == 0]['Body']

#spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam)
#non_spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_spam)

"""
plt.figure(figsize=(10, 6))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.title('spam word cloud')
plt.axis('off')
plt.show()


plt.figure(figsize=(10, 6))
plt.imshow(non_spam_wordcloud, interpolation='bilinear')
plt.title('no spam word cloud')
plt.axis('off')
plt.show()
"""
