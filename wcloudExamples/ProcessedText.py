import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Loading The Adventures of Sherlock Holmes by Arthur Conan Doyle
# from the Project Gutenberg
from urllib import request
url = "http://www.gutenberg.org/ebooks/1661.txt.utf-8"
response = request.urlopen(url)
rawText = response.read().decode('utf8')


# word cloud before text processing
wordcloud = WordCloud(max_font_size=72).generate(rawText)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# tokenizing
wordText = nltk.word_tokenize(rawText)

# Lemmatizing using POS tags
stop_words = set(stopwords.words('english'))  # stop words in English
lmt = WordNetLemmatizer()
wordPOS = nltk.pos_tag(wordText)
# removing punctuation marks & stop words, making all words lower case, 
wordPOSDePunct = [(w[0].lower(), w[1]) for w in wordPOS if w[0].isalpha()]
wordPOSNoStopwd = [w for w in wordPOSDePunct if w[0] not in stop_words]
# initializing the lammatized word list
wordPOSLemma = []
for wPair in wordPOSNoStopwd:
    if wPair[1][0] == 'J':   # i.e., adjectives
        wordPOSLemma.append(lmt.lemmatize(wPair[0],pos='a'))
    elif wPair[1][0] == 'V':  # i.e., verbs
        wordPOSLemma.append(lmt.lemmatize(wPair[0],pos='v'))
    elif 'RB' in wPair[1]:  # i.e., adverbs
        wordPOSLemma.append(lmt.lemmatize(wPair[0],pos='r'))
    else:
        wordPOSLemma.append(lmt.lemmatize(wPair[0]))

# concatenating all words into a long text
procText = ' '.join(wordPOSLemma)


# word cloud after text processing
wordcloud = WordCloud(max_font_size=72).generate(procText)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# just nouns only
wordNouns = []
for wPair in wordPOSNoStopwd:
    if 'NN' in wPair[1]:  # i.e., nouns
        wordNouns.append(lmt.lemmatize(wPair[0]))

# concatenating all words into a long text
nounText = ' '.join(wordNouns)

# word cloud after text processing
wordcloud = WordCloud(max_font_size=72).generate(nounText)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



# just verbs only
wordVerbs = []
for wPair in wordPOSNoStopwd:
    if 'VB' in wPair[1]:  # i.e., verbs
        wordVerbs.append(lmt.lemmatize(wPair[0],pos='v'))

# concatenating all words into a long text
verbText = ' '.join(wordVerbs)

# word cloud after text processing
wordcloud = WordCloud(max_font_size=72).generate(verbText)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()




# just adjectives only
wordAdjs = []
for wPair in wordPOSNoStopwd:
    if 'JJ' in wPair[1]:  # i.e., adjectives
        wordAdjs.append(lmt.lemmatize(wPair[0],pos='a'))

# concatenating all words into a long text
adjText = ' '.join(wordAdjs)

# word cloud after text processing
wordcloud = WordCloud(max_font_size=72).generate(adjText)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

