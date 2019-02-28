import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder


# loading the data
mushroomData = pd.read_csv('mushroom_toydata.csv')
feature_names = mushroomData.columns[1:]

# the target
LEEdible = LabelEncoder()
y = LEEdible.fit_transform(mushroomData.Edible)
Edible_class = LEEdible.classes_




# classifier, CapSurface
LECapSurface = LabelEncoder()
CapSurface = LECapSurface.fit_transform(mushroomData.CapSurface)
CapSurface_class = LECapSurface.classes_
dtCapSurface = DecisionTreeClassifier(max_depth=1,criterion='entropy',
                                      random_state=0)
dtCapSurface.fit(CapSurface.reshape(-1, 1),y)

# generating a decision tree
dot_data = export_graphviz(dtCapSurface, feature_names=feature_names[0:1],
                           class_names=Edible_class, 
                           filled=True, rounded=True,  
                           special_characters=True,
                           out_file=None) 
graph = graphviz.Source(dot_data)

### Only works on Jupyter notebook. Otherwise I have to create a separate file
graph





# classifier, CapColor
LECapColor = LabelEncoder()
CapColor = LECapColor.fit_transform(mushroomData.CapColor)
CapColor_class = LECapColor.classes_
dtCapColor = DecisionTreeClassifier(max_depth=1,criterion='entropy',
                                      random_state=0)
dtCapColor.fit(CapColor.reshape(-1, 1),y)

# generating a decision tree
dot_data = export_graphviz(dtCapColor, feature_names=feature_names[1:2],
                           class_names=Edible_class, 
                           filled=True, rounded=True,  
                           special_characters=True, 
                           out_file=None) 
graph = graphviz.Source(dot_data)

### Only works on Jupyter notebook. Otherwise I have to create a separate file
graph





# classifier, GillSize
LEGillSize = LabelEncoder()
GillSize = LEGillSize.fit_transform(mushroomData.GillSize)
GillSize_class = LEGillSize.classes_
dtGillSize = DecisionTreeClassifier(max_depth=1,criterion='entropy',
                                      random_state=0)
dtGillSize.fit(GillSize.reshape(-1, 1),y)

# generating a decision tree
dot_data = export_graphviz(dtGillSize, feature_names=feature_names[2:3],
                           class_names=Edible_class, 
                           filled=True, rounded=True,  
                           special_characters=True, 
                           out_file=None) 

graph = graphviz.Source(dot_data)

### Only works on Jupyter notebook. Otherwise I have to create a separate file
graph





# all three features
X = np.vstack([CapSurface, CapColor, GillSize]).T
dtAll = DecisionTreeClassifier(max_depth=3,criterion='entropy',
                                      random_state=0)
dtAll.fit(X,y)

# exporting a graphviz file
dot_data = export_graphviz(dtAll, feature_names=feature_names,
                           class_names=Edible_class, 
                           filled=True, rounded=True,  
                           special_characters=True, 
                           out_file=None) 

graph = graphviz.Source(dot_data)

### Only works on Jupyter notebook. Otherwise I have to create a separate file
graph
