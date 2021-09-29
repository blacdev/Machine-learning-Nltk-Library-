# This program performs sentimental analysis on movie reviews 

# Importing libraries

import nltk
import nltk.classify.util 
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus.reader import wordlist

# download the data
nltk.download("movie_reviews")

# define the function call to extract features

def extract_features (wordlist):
    return dict([(word,True) for word in wordlist])

if __name__ == '__main__':

    # loading positive and negative reviews

    positive_fileids = movie_reviews.fileids("pos")
    negative_fileids = movie_reviews.fileids("neg")

    features_positive =[(extract_features(movie_reviews.words(fileids = [f])),"positive") for f in positive_fileids]
    features_negative =[(extract_features(movie_reviews.words(fileids = [f])),"negative") for f in negative_fileids]

    # split the dataset into training and testing 

    threshold_factor = 0.8
    threshold_positive = int (threshold_factor * len (features_positive))
    threshold_negative = int (threshold_factor * len (features_negative))
    features_train = features_positive [:threshold_positive] + features_negative [:threshold_negative]
    features_test = features_positive [threshold_positive:] + features_negative [threshold_negative:]
    print("\n Number of Training datapoints: ", len (features_train))
    print("\n Number of Test datapoints: ", len (features_test))

    # Training the NaiveBayesClassifier

    classifier = NaiveBayesClassifier.train (features_train)
    print("\n Accuracy of the classifier: ", nltk.classify.util.accuracy (classifier, features_train))
    print("\nTop 10 informative words: ")
    for item in classifier.most_informative_features()[:10]:
        print(item[0])

    # input reviews

    input_reviews =[
        "This movie is fun",
        "The movie is very bad",
        "The movie is well constructed",
        "The movie is too poor",
        "Please dont watch",
        "It is a waste of time watching this movie"
        
    ]

    print("\n predictions: ")
    for review in input_reviews:
        print("\nReview: ", review)
        probdist = classifier.prob_classify (extract_features(review.split()))
        pred_sentiment = probdist.max()
        print ("Predicted Sentiment: ",pred_sentiment)
        print ("Probability: ", round(probdist.prob(pred_sentiment), 2))