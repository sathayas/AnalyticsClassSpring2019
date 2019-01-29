import pandas as pd

# loading the data
allData = pd.read_csv('AllAirlines_Mar2013.csv')

# total departing passengers for all airports
allDepart = allData.groupby('ORIGIN').sum().PASSENGERS.reset_index()
allDepart.columns = ['Airport','TotalDep'] # renaming the columns

# dataframe with Southwest airlines only
swData = allData[allData.UNIQUE_CARRIER == 'WN']

# total departing passengers for Southwest
swDepart = swData.groupby('ORIGIN').sum().PASSENGERS.reset_index()
swDepart.columns = ['Airport','SWDep'] # renaming the columns


# merging the two data frames
combinedDepart = pd.merge(allDepart, swDepart, on='Airport', how='outer')
combinedDepart.fillna(0, inplace=True)  # replacing NaN with 0


# Southwest's market share
combinedDepart['SWShare'] = combinedDepart['SWDep']/combinedDepart['TotalDep']
combinedDepart.fillna(0, inplace=True)  # replacing NaN with 0


# writing to a CSV file
combinedDepart.to_csv('SouthwestShare.csv',
                      columns=['Airport','SWShare'])



