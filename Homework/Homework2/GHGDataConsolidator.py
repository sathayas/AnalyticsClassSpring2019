numFiles = 2921
step = 20
sens = 5

fOut = open('GHG_Data.txt', 'w')

for iFile in range(1,numFiles,step):
    fName = '/Users/sh45474/Downloads/ghg_data/ghg.gid.site%04d' % iFile + '.dat'
    fIn = open(fName,'r')
    for iLine in range(sens):
        line = fIn.readline()
    fOut.write(line)
    fIn.close()

fOut.close()
