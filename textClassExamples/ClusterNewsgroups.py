import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from wordcloud import WordCloud

# loading the newsgroup data to be clustered
newsGroups = ['soc.religion.christian',
              'comp.graphics',
              'sci.med',
              'rec.sport.baseball']

from sklearn.datasets import fetch_20newsgroups
newsData = fetch_20newsgroups(subset='all',
                              categories=newsGroups,
                              shuffle=True,
                              random_state=0)

# news posts (X) and labels (Y)
X = newsData.data
Y = newsData.target
targetNames = newsData.target_names


# removing header lines from posts
headerTags = ['From:', 'Subject:', 'Organization:', 'Lines:',
              'Distribution:', 'Reply-To:', 'Host:', 'Keywords:',
              'Summary:', 'writes:']
def LineContains(text,tags):
    bTag = False
    for iTag in tags:
        if iTag in text:
            bTag = True
            break
    return bTag

X_NoHead = []
for iDoc in X:
    nonHeaderDoc = []
    Lines = iDoc.split('\n')
    for iLine in Lines:
        if not LineContains(iLine,headerTags):
            nonHeaderDoc.append(iLine)
    X_NoHead.append('\n'.join(nonHeaderDoc))


# tokenize, lower case, remove punctuation, remove stop words
stop_words = set(stopwords.words('english'))  # stop words in English
ps = PorterStemmer()  # stemmer object
X_proc =[]
for iDoc in X_NoHead:
    # tokenize into words
    wordText = nltk.word_tokenize(iDoc)
    # removing punctuation marks & stop words, making all words lower case,
    wordDePunct = [w.lower() for w in wordText if w.isalpha()]
    wordNoStopwd = [w for w in wordDePunct if w not in stop_words]
    # stemming
    wordStem = [ps.stem(w) for w in wordNoStopwd]
    # putting back into a document
    X_proc.append(' '.join(wordStem))


# converting to frequencies to be used as features
X_counts = CountVectorizer().fit_transform(X_proc)
X_tf = TfidfTransformer().fit_transform(X_counts)


# K-means clustering
km = KMeans(n_clusters=4)
km.fit(X_tf)  # fitting the principal components
Y_clus = km.labels_   # clustering info resulting from K-means


# the performance of clustering
print('ARI=%6.4f' % adjusted_rand_score(Y, Y_clus))
print('AMI=%6.4f' % adjusted_mutual_info_score(Y, Y_clus))



# generating word cloud for each cluster
for iClus in range(max(Y_clus)+1):
    # first, concatenating all texts for that cluster
    allText = ''
    for j,jClus in enumerate(Y_clus):
        if jClus==iClus:  # i.e., text belongs in the cluster
            allText += X_proc[j]
            allText += ' '

    # genrating the word cloud

    wordcloud = WordCloud(max_font_size=100,
                          background_color='white',
                          collocations=False,
                          width=800,
                          height=400).generate(allText)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Cluster ' + str(iClus+1), fontsize=18)
    plt.show()
