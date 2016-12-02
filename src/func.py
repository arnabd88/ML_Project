
import sys
import re
import copy
import math
import numpy
import os

def trimStr( str1 ):
	l2 = ''
	for i in str1:
		if(i!=' '):
			l2 = l2+i
	return l2



def NormalizeScore(FScore):
	Tools = FScore[0].split(":")[1].split()
	Score = dict(dict([]))
	for t in Tools:
		Score[t] = dict([])
	for i in range(1,len(FScore)):
		catName = trimStr(FScore[i].split(":")[0])
		scoreList = FScore[i].split(":")[1].split()
		tempScore = []
		tempActualScore = []
		for j in range(0,len(scoreList)):
			if (scoreList[j] != "*"):
				tempScore.append(numpy.absolute(int(scoreList[j])))
				tempActualScore.append(int(scoreList[j]))
		SumScore = sum(tempScore)
		MaxScore = numpy.amin(numpy.array(tempScore))
		MinScore = numpy.amin(numpy.array(tempActualScore))
		for j in range(0,len(scoreList)):
			if(scoreList[j] != "*"):
				Score[Tools[j]][catName] = float(scoreList[j])/SumScore
			else:
				Score[Tools[j]][catName] = (float(MinScore)/SumScore) - (float(MaxScore)/SumScore)
	return Score
			
			
def decomposeBM(BMList):
	BMDict = dict([])
	for line in BMList:
		splitList = line.split(":")
		fstr = splitList[1].split("[")[1].split("]")[0]
		flist = fstr.split(",")
		BMDict[splitList[0]] = flist
	return BMDict


def reverseBMSubList(BMSubList):
	BMSubDict = dict([])
	for line in BMSubList:
		CatName = trimStr(line.split(":")[0])
		SubCatList = line.split(":")[1].split(",")
		for i in SubCatList:
			BMSubDict[i] = CatName
	return BMSubDict
		


def WriteTrainingData(genData, Score, DPath):
	d = DPath+"/Training/"
	if not os.path.exists(d):
		os.makedirs(d)
	for t in Score.keys():
		tfile = open(d+"/"+t+".data", 'w+')
		print len(genData.keys())
		count = 0
		for bm in genData.keys():
			strdata = ""
			strdata = strdata+str(Score[t][genData[bm]['Category']])
			strdata = strdata+" "+" ".join(genData[bm]['feature'])
			tfile.write(strdata+"\n")
			count = count + 1
		print count
		tfile.close()
	
			
	
	


def generateTrainingData( BMList, BMSubList, FScore, DPath):
	currDir = os.getcwd()
	Tools = FScore[0].split(":")[1].split()
	#print Tools
	##------- Reverse the BMSubList ----------------
	BMSubDict = reverseBMSubList(BMSubList)
	##----------------------------------------------
	Score = NormalizeScore(FScore)
	##------- BreakDown the BenchMarkList ----------
	BMDict = decomposeBM(BMList)
	##----------------------------------------------
	#print DPath
	floops = open(DPath+"/loops/loops_metrics", 'r').read().splitlines()
	fvars  = open(DPath+"/vars/roles_metrics", 'r').read().splitlines()
	##----- work on the vars first -----
	genData = dict(dict([]))
	for line in fvars[1:]:
		tempDict = dict([])
		str1 = line.split()
		str2 = str1[0].split("/")
		subcat = str2[-2]
		#name = str2[-1].split(".")[0]
		name = ".".join(str2[-1].split(".")[:-1])
		tempDict['feature'] = str1[1:]
		tempDict['Category'] = BMSubDict[subcat]
		genData[name] = tempDict
	
	for line in floops[1:]:
		str1 = line.split()
		str2 = str1[0].split("/")
		subcat = str2[-2]
		name = ".".join(str2[-1].split(".")[:-1])
		if(genData[name]['Category'] != BMSubDict[subcat]):
			#print genData[name]['Category'], BMSubDict[subcat], name
			print "Err: Subcat mismatch in var and loop metrics"
			exit(-1)
		else:
			genData[name]['feature'] = genData[name]['feature']+str1[1:]

	WriteTrainingData(genData,Score,DPath)
		
		
