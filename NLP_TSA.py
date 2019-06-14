# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('tsa.csv',header=0)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 16000):
    tweet = re.sub('[^a-zA-Z]', ' ', dataset['SentimentText'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values[0:16000]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




# Saving the classifier
import pickle
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

save_vectorizer = open("naivebayes_vectorizer.pickle","wb")
pickle.dump(cv, save_vectorizer)
save_vectorizer.close()





# Predicting the new data
tweetPredict = re.sub('[^a-zA-Z]', ' ', "Your Tweet here")
tweetPredict = tweetPredict.lower()
tweetPredict = tweetPredict.split()
ps = PorterStemmer()
tweetPredict = [ps.stem(word) for word in tweetPredict if not word in set(stopwords.words('english'))]
tweetPredict = ' '.join(tweetPredict)
prediction = cv.transform([tweetPredict]).toarray()
classifier.predict(prediction)
