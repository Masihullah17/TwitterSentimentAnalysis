import pandas as pd
from textblob import TextBlob

data = pd.read_csv('tsa.csv',header=0)

i = 0
for tweet in data['SentimentText']:
    print(tweet)
    
    #Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet)
    print(analysis.sentiment)
    print("")

    i  = i + 1

    if i > 10: #To stop after analysing 10 tweets
        break;
