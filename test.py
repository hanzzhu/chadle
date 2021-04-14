import datetime
import os

# ProjectNames = next(os.walk(RootDir))[1]


project_list = ['abc', 'AAs']
ProjectName = 'a'
var = list(x for x in project_list if 'A' in x)
print(var[0])

var = list((x for x in list(map(str.upper, project_list)) if ProjectName.upper() in x))
print(var)
# print(ProjectNames)
