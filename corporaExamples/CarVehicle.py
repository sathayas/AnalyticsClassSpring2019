import nltk
from nltk.corpus import wordnet as wn

# Generating synsets
synsCar = wn.synsets('car')
synsVehicle = wn.synsets('vehicle')

# loop over different meanings
maxSim = 0
for iCar in synsCar:
    for iVeh in synsVehicle:
        simScore = iCar.wup_similarity(iVeh)
        if simScore>maxSim:
            maxSim = simScore
            maxCombo = [iCar, iVeh]

# printing the max similarity
print('Maximum similarity: %6.4f' % maxSim)

# printing out the definitions
print(str(maxCombo[0]) + ': ' + str(maxCombo[0].definition()))
print(str(maxCombo[1]) + ': ' + str(maxCombo[1].definition()))
