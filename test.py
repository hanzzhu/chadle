import os

ProjectDict = ['Animals', 'NTBW Image Analytics']
RootDir = 'C:/Users/930415/Desktop/Chadle_Data'
testDir = "C:/Users/930415/Desktop/Chadle_Data/Animals"

ProjectNames = next(os.walk(RootDir))[1]


print(ProjectNames)