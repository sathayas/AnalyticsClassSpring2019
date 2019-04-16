import nltk

# loading the Emma by Jane Austen
from nltk.corpus import gutenberg
emmaRawText = gutenberg.raw('austen-emma.txt')

# tokenizing
emmaSents = nltk.sent_tokenize(emmaRawText)
print(emmaSents[5])
emmaWords = nltk.word_tokenize(emmaSents[5])

# POS tagging of an example sentence
emmaTagged = nltk.pos_tag(emmaWords)
print(emmaTagged)

# extracting verbs only (starting with VB)
emmaVerbs = [w for w in emmaTagged if 'VB' in w[1]]
print(emmaVerbs)

# extracting adverbs only
emmaAdv = [w for w in emmaTagged if 'RB' in w[1]]
print(emmaAdv)

# extracting proper nouns only
emmaNNP = [w for w in emmaTagged if 'NNP' in w[1]]
print(emmaNNP)
