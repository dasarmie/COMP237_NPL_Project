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
from sklearn.metrics import confusion_matrix
import numpy as np
import re
np.random.seed(6)
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
print(label_counts)
plt.figure(figsize=(6, 5))
plt.bar(label_counts.index, label_counts.values, color=['green', 'red'])
plt.title('Label distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(label_counts.index, ['No spam', 'Spam'], rotation=0)
plt.show()

#prepare data for modeling
input_data.drop(index=range(400,640), inplace=True)
input_data = input_data.sample(frac=1)

#function to remove special characters
def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s.$#%,!?:@()\n\t]'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

#this code block generates word cloud before applying the remove_special_characters function
text = ' '.join(input_data['Body'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
#applying the function
input_data['Body'] = input_data['Body'].apply(lambda x: remove_special_characters(x))
#same codeblock to generate word cloud after applying the function
text = ' '.join(input_data['Body'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

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
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['No spam', 'Spam']
plot_confusion_matrix(cm)
plt.show()
#manually testing the model

new_emails = [
        "You have an upcoming meetup scheduled at Toronto, Ontario.\
        To unsubscribe from this group, send an email to:",

        "Hi all.\
        Does anyone know how to set up dual monitors? one is actually a stand-alone\
        on a Mac?\
        ",

        "Hi, I'm looking to build a completely silent pc. It's gonna be a gateway for a wireless network and will sit in my room (as my room is only spitting distance from the chimney where i'll be mounting the aerial)"
        ,

        "I will be out of the office starting  04/12/2024 and will not return until 04/20/2024. I am out of the office until Tuesday 20th April.   I will reply to messages on my return.Thank you."
        ,

        "We have been trying to reach you.\
        Please see this new offer we have developed for your personalized interest.\
        Only for this month, it is upto 80% free.\
        Just go to our website and click Buy Now\
        ",

        "Hi, This is a reminder for your meeting with your lawyer. Kindly confirm your availability for the meeting. \
        Seats are filling soon. Please respond as soon as possible.\
        "

]

X_new_emails = vectorizer.transform(new_emails)
X_new_tfidf = tfidf_transformer.transform(X_new_emails)

majority_class = input_data[input_data['Label'] == 0]
print(majority_class['Body'])

y_new_pred = classifier.predict(X_new_tfidf)

for email, predicted_label in zip(X_new_emails, y_new_pred):
    print(f"email: \n{email}")
    print(f"Predicted Label: {predicted_label} {'Spam' if predicted_label==1 else 'Not Spam'}")
