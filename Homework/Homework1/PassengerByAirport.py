import pandas as pd
import matplotlib.pyplot as plt

# loading the data into a data frame
swData = pd.read_csv('Southwest_Mar2013.csv')

# calculating the total departing and arriving passengers
departPass = swData.groupby('ORIGIN').sum().reset_index()
departPass.columns = ['Airport', 'TotalDep']  # renaming the columns
arrivePass = swData.groupby('DEST').sum().reset_index()
arrivePass.columns = ['Airport', 'TotalArr']  # renaming the columns

# merging them together
totalPass = pd.merge(departPass, arrivePass, on='Airport', how='outer')
totalPass.fillna(0, inplace=True)  # replacing NaN with 0

# saving to a CSV file
totalPass.to_csv('NumPassengerByAirport.csv', index=False)



# plotting
plt.plot(totalPass.TotalDep, totalPass.TotalArr, 'r.')
plt.xlabel('Total departing passengers')
plt.ylabel('Total arriving passengers')
plt.show()



# top 5 airports with total departing passengers
print(totalPass.sort_values(by='TotalDep', ascending=False).head(5))


