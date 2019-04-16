import nltk

# loading the Emma by Jane Austen
from nltk.corpus import gutenberg
emmaRawText = gutenberg.raw('austen-emma.txt')

# tokenizing
emmaSents = nltk.sent_tokenize(emmaRawText)
emmaWords = nltk.word_tokenize(emmaSents[5])

# POS tagging of an example sentence
emmaTagged = nltk.pos_tag(emmaWords)


# extracting various POS tags
emmaVB = [w for w in emmaTagged if 'VB' in w[1]]
emmaNN = [w for w in emmaTagged if 'NN' in w[1]]
emmaJJ = [w for w in emmaTagged if 'JJ' in w[1]]
emmaRB = [w for w in emmaTagged if 'RB' in w[1]]
print('Verbs (VB): ', len(emmaVB))
print('Nouns (NN): ', len(emmaNN))
print('Adjectives (JJ): ', len(emmaJJ))
print('Adverbs (RB): ', len(emmaRB))


