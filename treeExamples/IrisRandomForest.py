import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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



# consolidating the features
X = np.vstack([xCapShape,xCapSurface,xCapColor]).T

# spliting the data into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000,
                                                    random_state=2000)

# random forest classifier, training & testing
rf = RandomForestClassifier(criterion='entropy',
                            n_estimators = 50,
                            min_samples_leaf = 9,
                            max_depth = 7,
                            random_state=0)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=Edible_class))



# For a comparision, 
# decision tree classifier, training & testing
dt = DecisionTreeClassifier(criterion='entropy', 
                            min_samples_leaf = 9,
                            max_depth = 7,
                            random_state=0)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,
                            target_names=Edible_class))
