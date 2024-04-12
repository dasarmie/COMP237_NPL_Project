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
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import numpy as np
import re
np.random.seed(66)
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
# plt.figure(figsize=(6, 5))
# plt.bar(label_counts.index, label_counts.values, color=['green', 'red'])
# plt.title('Label distribution')
# plt.xlabel('Label')
# plt.ylabel('Count')
# plt.xticks(label_counts.index, ['No spam', 'Spam'], rotation=0)
# plt.show()

#prepare data for modeling
input_data = input_data.sample(frac=1)

#function to remove special characters
def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s$#-%]'  # Matches any character that is not alphanumeric or whitespace
    pattern += r'|[\n]'  # Matches newline characters (\n)
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

#this code block generates word cloud before applying the remove_special_characters function
text = ' '.join(input_data['Body'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
#applying the function
input_data['Body'] = input_data['Body'].apply(lambda x: remove_special_characters(x))
#same codeblock to generate word cloud after applying the function
text = ' '.join(input_data['Body'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

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

print("Confusion matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)
#manually testing the model
# new_emails = [
#     "Hi, you won 500000 please enter credit card number.",
#     "Hello, This is a friendly reminder to pay your credit card balance by 21 of september",
#     "Hi, Thanks for your email.",
#     "Congratulations! You've won a free cruise to the Bahamas. Click here to claim your prize!"
# ]
new_emails = [
            "Do\
        you Have a new House?Yes? Then we can SAVE\
        you $$$ Money!\
Our High Quality roofing system & HVAC\
        comes with a money-back guarantee, a 1-year\
        Warranty*, and get FREE SHIPPING\
        on ALL orders!\
and best of all...They Cost\
         30-70%  Less        than\
        Retail Price!*Click here\
        to visit Our Website!*or Call us Toll-Free @\
        1-800-758-8084!\
              ",

        "Thank You,Your payment was obtained from a purchased\
        credit card, Reference # 1200-000-1.If you wish to unsubscribe from this list, please\
        Click here and enter your \
        reference number and credit card details into the box. If you have previously \
        unsubscribed and are still receiving this message, you may email our \
        Credit Card department, \
        or call 1-888-123-4567, or write us at: Payment Team, 35 Progress Avenue, \
        Toronto, ON, M1A9A9. \
        2024 Web Credit Inc. All Rights Reserved.",

        "Dear, I hope this email finds you well.\
        Please take a look at the attachment that you requested.\
        Looking forward to hear from you soon.",

        "Hello,\
        You have a new payment awaiting on your mortgage. \
        Kindly get back to us with this email address.\
        ",

        "We have been trying to reach you.\ "
        "Please see this new offer we have developed for your personalized interest.\
        Only for this month, it is upto 80% free.\
        Just go to our website and click Buy Now\
        ",

        "Hi, This is a reminder for your meeting with your lawyer. Kindly confirm your availability for the meeting. \
        Seats are filling soon. Please respond as soon as possible.\
        "

]

X_new_emails = vectorizer.transform(new_emails)
X_new_tfidf = tfidf_transformer.transform(X_new_emails)



y_new_pred = classifier.predict(X_new_tfidf)

for email, predicted_label in zip(X_new_emails, y_new_pred):
    print(f"email: \n{email}")
    print(f"Predicted Label: {predicted_label} {'Spam' if predicted_label==1 else 'Not Spam'}")
