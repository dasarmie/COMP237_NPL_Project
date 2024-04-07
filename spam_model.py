"""
NLP project to classify spam messages using Naive Bayes Classifier
Group 6 - W24
Members: Helia Mozaffari, Susmita Roy, Sandeep Neupane, Diego Sarmiento
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#1.Load the data into a pandas data frame.
path = './src/Data6.csv'
input_data = pd.read_csv(path)

#2.Carry out some basic data exploration and present your results.
print(input_data.info())
print(input_data.head())
print(input_data.tail())
print(input_data.shape)
# Data distribution
label_counts = input_data['Label'].value_counts()
plt.figure(figsize=(6, 5))
plt.bar(label_counts.index, label_counts.values, color=['green', 'red'])
plt.title('Label distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(label_counts.index, ['No spam', 'Spam'], rotation=0)
plt.show()

#prepare data for modeling
input_data = input_data.sample(frac=1)
vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(input_data['Body'])
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_count)
y = input_data['Label']
train_size = int(0.75 * len(input_data))

X_train = X_tfidf[:train_size]
X_test = X_tfidf[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
print(input_data.Label)

#train using naive bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

scores = cross_val_score(classifier, X_train, y_train, cv=5)
print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


#manually testing the model
new_emails = [
    "Hi, you won 500000 please enter credit card number.",
    "Hello, This is a friendly reminder to pay your credit card balance by 21 of september",
    "Hi, Thanks for your email.",
    "Congratulations! You've won a free cruise to the Bahamas. Click here to claim your prize!"
]

X_new_emails = vectorizer.transform(new_emails)
X_new_tfidf = tfidf_transformer.transform(X_new_emails)



y_new_pred = classifier.predict(X_new_tfidf)

for email, predicted_label in zip(X_new_emails, y_new_pred):
    print(f"email: {email}")
    print(f"Predicted Label: {predicted_label}")
