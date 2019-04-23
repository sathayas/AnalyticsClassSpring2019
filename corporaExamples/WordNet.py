import nltk
from nltk.corpus import wordnet as wn

# synset
syns = wn.synsets('program')
print(syns)

# definition for each meaning
for isyn in syns:
    print(isyn, ':', end='')
    print(isyn.definition())

# synnonyms for the computer program
print(wn.synset('program.n.07').lemma_names())

# synnonyms for a broadcasting program
print(wn.synset('broadcast.n.02').lemma_names())


# another synset
syns = wn.synsets('plan')
for isyn in syns:
    print(isyn, ':', end='')
    print(isyn.definition())

# synonyms
print(syns[0].lemma_names())

# examples
print(syns[0].examples())


# program and code
print(wn.synset('program.n.07').definition())
print(wn.synset('code.n.03').definition())

# similarity
synsProgram = wn.synset('program.n.07')
synsCode = wn.synset('code.n.03')
print(synsProgram.wup_similarity(synsCode))


# program as in TV or radio show
print(wn.synset('broadcast.n.02').definition())

synsBroadcast = wn.synset('broadcast.n.02')
synsCode = wn.synset('code.n.03')
print(synsBroadcast.wup_similarity(synsCode))
