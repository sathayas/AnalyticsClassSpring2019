import nltk
from nltk.corpus import PlaintextCorpusReader

# creating own corpus
corpus_root = 'SentimentReviews'
reviews = PlaintextCorpusReader(corpus_root, ".*\.txt")

# contents of the corpus
print(reviews.fileids())
print(reviews.raw('yelp_labelled.txt')[:300])
print(reviews.words('yelp_labelled.txt')[:80])

