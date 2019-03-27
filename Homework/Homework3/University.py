import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# function to convert string numbers and ranges to numbers
# for ranges, it returns the middle value
# for ranges with upper bound undefined, convert to lower bound
# for ratios, it returns the fraction
def convert_num(s):
    s = s.strip()
    if 'act-' in s: # act score
        x = np.nan
    elif 'ratio' in s:    # if ratio
        s = s.replace('ratio:','').replace('ration:','')
        xs = s.split(':')
        if float(xs[1])==0:  # if the denominator is zero
            x = 10.0
        else:
            x = float(xs[0])/float(xs[1])
    else:
        ss = s.split(':')[-1]
        if '-' not in ss and '+' not in s:  # if just a single number
            x = float(ss)
        elif '-' in ss:   # if range
            xs = ss.split('-')
            if xs[1]=='':
                x = float(xs[0])
            else:
                x = (float(xs[0])+float(xs[1]))/2
        else:  # otherwise its the lower bound
            x = float(ss.replace('+',''))
    return x


# function to skip lines, read a single record, then returns a dictionary
def read_record():
    b_eof = False   # end of file indicator

    # skip until you hit the beginning of a record
    while True:
        line = f.readline().lower()
        if '(def-instance' in line:  # beginning of the record
            break
        if '=====' in line:   # end of the data file
            b_eof = True
            break

    # initializing the data storage
    univDict = {}
    # read lines only if not the end of file
    if not b_eof:

        # keep reading the record until the end of the record
        lineData = []
        while True:
            if '%' not in line:  # if not missing value
                # remove leading and trailing spaces, remove praentheses
                lineData.append(line.strip().replace('(','').replace(')',''))
            if '))' in line or line==')\n':   # you hit the last line of the record
                break
            line = f.readline().lower()
            
        # converting the line data into a dictionary
        for iData in lineData:
            if len(iData)>0 and 'academic-emphasis' not in iData:
                listSubLine = iData.split(' ')
                univKey = listSubLine[0].replace(':','-')  # the key
                univValue = listSubLine[-1]   # the value
                # if numberic data, then returns a number. Otherwise string is returned
                if any(char.isdigit() for char in univValue) and univKey!='def-instance':
                    univDict[univKey] = convert_num(univValue)
                else:
                    univDict[univKey] = univValue
    return univDict, b_eof
    

###### loading the data
f = open('university_data.txt','r')
univData = []
while True:
    uDict, bEnd = read_record()
    if bEnd:
        break
    else:
        if not uDict['def-instance'][-1].isnumeric():  # only records non-duplicates
            univData.append(uDict)
f.close()

# converting dictionaries to data frame
univDataDF = pd.DataFrame(univData)
# then shuffling
univDataDF = univDataDF.sample(frac=1).reset_index(drop=True)  



###### data cleaning
# removing columns with many missing values
univDataDF.drop(['colors','mascot','religious-backing'],axis=1,inplace=True)
# removing incomplete observations
univDataDF.dropna(inplace=True)
# creating target, private or public
y = []
for iRow in range(len(univDataDF)):
    if 'private' in univDataDF.iloc[iRow].control:
        y.append(1)
    else:
        y.append(0)
univDataDF['y'] = y
target_labels = ['public','private']




###### feature selection
# features, categorical and continuous
LELoc = LabelEncoder().fit_transform(univDataDF.location)
LEState = LabelEncoder().fit_transform(univDataDF.state)
xCat = np.vstack([LELoc, LEState]).T
xCont = univDataDF[['academics',
                    'expenses',
                    'male-female',
                    'no-applicants',
                    'no-of-students',
                    'percent-admittance',
                    'percent-enrolled',
                    'percent-financial-aid',
                    'quality-of-life',
                    'sat',
                    'social']]


# categorical features
chiStat, chiP = chi2(xCat,y)
print(chiP)
#
# Feature to include:
#    state
#

# continuous features
fStat, fP = f_classif(xCont,y)
print(fP)
#
# Features to include:
#    academics
#    expenses
#    no-of-students
#    percent-admittance
#    percent-enrolled
#    percent-financial-aid
#    sat
#



###### final feature array
# I use one hot encoder -- a collection of dummy variables for state
XCat = OneHotEncoder().fit_transform(LEState.reshape(-1,1)).toarray()
# continuous features
contMatrix = univDataDF[['academics',
                         'expenses',
                         'no-of-students',
                         'percent-admittance',
                         'percent-enrolled',
                         'percent-financial-aid',
                         'sat']]
XCont = np.array(contMatrix)
X = np.hstack([XCont, XCat])




###### SVM classifier
# parameters for grid search
param = {'classify__C': [10,1,0.1],
         'classify__kernel': ['rbf','poly','linear']}

# pipeline of transformations
svm = Pipeline([('normalize',StandardScaler()),
                ('classify',SVC())])

# grid search
grid_svm = GridSearchCV(svm, param_grid=param, cv=5)
grid_svm.fit(X,y)

# winner
print(grid_svm.best_params_)
print(grid_svm.best_score_)




####### now the feature array has no expenses
# I use one hot encoder -- a collection of dummy variables for state
XCat = OneHotEncoder().fit_transform(LEState.reshape(-1,1)).toarray()
# continuous features
contMatrix = univDataDF[['academics',
                         'no-of-students',
                         'percent-admittance',
                         'percent-enrolled',
                         'percent-financial-aid',
                         'sat']]
XCont = np.array(contMatrix)
X = np.hstack([XCont, XCat])




###### SVM classifier
# parameters for grid search
param = {'classify__C': [10,1,0.1],
         'classify__kernel': ['rbf','poly','linear']}

# pipeline of transformations
svm = Pipeline([('normalize',StandardScaler()),
                ('classify',SVC())])

# grid search
grid_svm = GridSearchCV(svm, param_grid=param, cv=5)
grid_svm.fit(X,y)

# winner
print(grid_svm.best_params_)
print(grid_svm.best_score_)
