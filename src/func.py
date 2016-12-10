
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

def reportTrend(simdata,phase):
	winTracker = [(value[1],key) for key,value in simdata.items()]
	winTracker.sort()
	print "########## Result During ",phase," phase ###############"
	for j in range(1,len(winTracker)+1):
		print "Position:", j,",Tool:",winTracker[-j][1], ",Score:",winTracker[-j][0]
	



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
				tempScore.append(numpy.absolute(float(scoreList[j]))) ##tracks the absolute value
				tempActualScore.append(float(scoreList[j])) ## tracks the int value
		SumScore = sum(tempScore)
		MaxScore = numpy.amax(numpy.array(tempScore))
		MinScore = numpy.amin(numpy.array(tempActualScore))
		for j in range(0,len(scoreList)):
			if(scoreList[j] != "*"):
				Score[Tools[j]][catName] = float(scoreList[j])/SumScore
			else:
				Score[Tools[j]][catName] = (float(MinScore)/SumScore) - numpy.absolute((float(MaxScore)))
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

def getBMSubDict(BMSubList):
	BMSubDict = dict([])
	for line in BMSubList:
		CatName = trimStr(line.split(":")[0])
		SubCatList = line.split(":")[1].split(",")
		BMSubDict[CatName] = SubCatList
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
	#print Score
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
	return [genData,Score]
		
	
def parseInfo ( xvecs):
	dsize = len(xvecs)
	xdata = []
	dtemp = []
	ylabel = []
	for i in range(0,dsize):
		line = xvecs[i].split()
		dtemp = [1.0]  ## Adding the bias term
		for j in range(1,len(line)):
			dtemp.append(float(line[j]))
		xdata.append(dtemp)
		ylabel.append(float(line[0]))
	return [xdata, ylabel]

def genAllTestData(tpath,tools):
	xdata = []
	ydata = []
	for t in tools:
		filedata = open(tpath+t+".data", "r+").read().splitlines()
		[txdata, tydata] = parseInfo(filedata)
		xdata = xdata+txdata
		ydata = ydata+tydata
	return [xdata,ydata]
		
		

def getPredictedScoreError(xdata, ydata, wvec):
	wtxSum = 0.0
	errorTrack = []
	for i in range(0,len(xdata)):
		#print wvec
		wtx = numpy.dot(wvec, xdata[i])
		errorTrack.append([ydata[i]-wtxSum])
		#print "Value Mismatch = ", ydata[i]-wtx, ydata[i], wtx
		wtxSum += wtx
	return [errorTrack, wtxSum]

def getPredictedScoreError2(xdata, ydata, wvec):
	wtxSum = 0.0
	errorTrack = []
	for i in range(0,len(xdata)):
		#print len(wvec) ,len(xdata[i])
		wtx = numpy.dot(wvec, xdata[i])
		errorTrack.append([ydata[i]-wtxSum])
		#print "Value Mismatch = ", ydata[i]-wtx, ydata[i], wtx
		wtxSum += wtx
	return [errorTrack, wtxSum]


def genData(trainId, testId):
	genDataTrain = {}
	genDataTest  = {}
	if(trainId != -1):
		Dpath     = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
		print Dpath+sys.argv[trainId + 2]
		BMList    = open(Dpath+sys.argv[trainId + 2], 'r+').read().splitlines()
		BMSubList = open(Dpath+sys.argv[trainId + 3], 'r+').read().splitlines()
		FScore    = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()
		[genDataTrain,scoreTrain] = generateTrainingData(BMList, BMSubList,FScore,Dpath)
	if(testId != -1):
		Tpath     = os.getcwd()+"/../Data/"+sys.argv[testId + 1]+"/"
		print Tpath+sys.argv[testId + 2]
		TBMList    = open(Tpath+sys.argv[testId + 2], 'r+').read().splitlines()
		TBMSubList = open(Tpath+sys.argv[testId + 3], 'r+').read().splitlines()
		TFScore    = open(Tpath+sys.argv[testId + 4], 'r+').read().splitlines()
		[genDataTest,scoreTest] = generateTrainingData(TBMList, TBMSubList,TFScore,Tpath)
	return [[genDataTrain,scoreTrain], [genDataTest,scoreTest]]


def getCategoryData(genDataStruct, cat, subcat):
	xdata = []
	#print genDataStruct[genDataStruct.keys()[0]]
	#print genDataStruct.keys()[0]
	for key in genDataStruct.keys():
		if(genDataStruct[key]['Category'] == cat):
			xdata.append([1.0]+map(float, genDataStruct[key]['feature']))


	return xdata

