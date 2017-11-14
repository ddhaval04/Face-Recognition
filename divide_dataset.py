from shutil import copyfile
import random
import os

# Location where the entire dataset is present
Faces = ['F:\\Data Science\\DL Project\\images\\Dhruv\\', "F:\\Data Science\\DL Project\\images\\Janvi\\","F:\\Data Science\\DL Project\\images\\Karan\\"]

# Locations where the divided dataset is being saved
Test = ["F:\\Data Science\\DL Project\\data\\test\\Dhruv\\","F:\\Data Science\\DL Project\\data\\test\\Janvi\\","F:\\Data Science\\DL Project\\data\\test\\Karan\\"]

Validation = ["F:\\Data Science\\DL Project\\data\\validation\\Dhruv\\","F:\\Data Science\\DL Project\\data\\validation\\Janvi\\","F:\\Data Science\\DL Project\\data\\validation\\Karan\\"]

Training = ["F:\\Data Science\\DL Project\\data\\training\\Dhruv\\", "F:\\Data Science\\DL Project\\data\\training\\Janvi\\", "F:\\Data Science\\DL Project\\data\\training\\Karan\\"]

FileList = [0] * 3
for i in xrange(3):
    FileList[i] = os.listdir(Faces[i])
    random.shuffle(FileList[i])

for i in xrange(3):
    path = Test[i]
    if not os.path.exists(path):
        os.mkdir(path)
    path = Validation[i]
    if not os.path.exists(path):
        os.mkdir(path)
    path = Training[i]
    if not os.path.exists(path):
        os.mkdir(path)

for i in xrange(len(FileList)):
    leng = int(len(FileList[i]))
    for j in xrange(leng):
	if(j<leng*0.6):	
            copyfile(Faces[i]+ FileList[i][j], Training[i] + FileList[i][j])
	elif(j<0.8*leng):
	    copyfile(Faces[i] + FileList[i][j], Test[i] + FileList[i][j])
	else:
	    copyfile(Faces[i] + FileList[i][j], Validation[i] + FileList[i][j])