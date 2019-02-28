import numpy as np
import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# loading the data
mushroomData = pd.read_csv('mushroom_cap.csv')
feature_names = mushroomData.columns[1:]


### Converting string categorical variables to numerical categorical

# the target
LEEdible = LabelEncoder()
y = LEEdible.fit_transform(mushroomData.Edible)
Edible_class = LEEdible.classes_
# y=0: edible
# y=1: poisonous

# CapShape
LECapShape = LabelEncoder()
xCapShape = LECapShape.fit_transform(mushroomData.CapShape)
CapShape_class = LECapShape.classes_
# xCapShape=0: bell
# xCapShape=1: conical
# xCapShape=2: convex
# xCapShape=3: flat
# xCapShape=4: knobbed
# xCapShape=5: sunken

# CapSurface
LECapSurface = LabelEncoder()
xCapSurface = LECapSurface.fit_transform(mushroomData.CapSurface)
CapSurface_class = LECapSurface.classes_
# xCapSurface=0: fibrous
# xCapSurface=1: grooves
# xCapSurface=2: scaly
# xCapSurface=3: smooth

# CapColor
LECapColor = LabelEncoder()
xCapColor = LECapColor.fit_transform(mushroomData.CapColor)
CapColor_class = LECapColor.classes_
# xCapColor=0: brown
# xCapColor=1: buff
# xCapColor=2: cinnamon
# xCapColor=3: gray
# xCapColor=4: green
# xCapColor=5: pink
# xCapColor=6: purple
# xCapColor=7: red
# xCapColor=8: white
# xCapColor=9: yellow



#### Exercise code here!

