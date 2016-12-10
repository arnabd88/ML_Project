import csv
import sys
import re
import copy
import math
import numpy
import os

##--- import app libs ---##
import func
import linearRegression

genId   = -1
trainId = -1
learnId = -1
testId  = -1

Dpath     = ""
BMList    = []
BMSubList = []
FScore    = []

if("-trainDir" in sys.argv):
	trainId = sys.argv.index("-trainDir")
else:
	print "No Training. Use already trained weights"

if("-gen" in sys.argv):
        genId = sys.argv.index("-gen")

if("-learn" in sys.argv):
        learnId = sys.argv.index("-learn")
if("-testDir" in sys.argv):
		testId = sys.argv.index("-testDir")
        

#if(genId != -1):
[[genDataTrain,scoreTrain], [genDataTest,scoreTest]] = func.genData(trainId, testId)

##------ Train on every tool -------##
w = {}
if(learnId != -1):
        Tools = []
	
        Dpath  = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
      	FScore = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()
        Tools  = FScore[0].split(":")[1].split()

        #w = {} #[tool -> weight([index -> val])]
        TrainToolsDict = {}
        for tool in Tools:
             ##--------- Get the training data ---------##
             trainFileName = Dpath+"/Training/"+tool+".data"

             trainData = csv.reader(open(trainFileName, "rb"), delimiter=" ")

             w_temp = linearRegression.LinearRegression(trainData)
             w[tool] = w_temp
			 ##---- Get the error norm and score on the training data -----##
             trainData = open(trainFileName, "rb").read().splitlines()
             [xdata,ydata] = func.parseInfo(trainData)
             TrainToolsDict[tool] = func.getPredictedScoreError(xdata,ydata, w_temp)

        func.reportTrend(TrainToolsDict, "Training on SVComp14")

     #   print "w:" , w

##------ Testing on SVComp15 (Predict the overall Winner)----------
if(testId!=-1 and trainId!=-1): ##indicates training dataStructure is available
	
	TestToolDict = {}
	Tpath = os.getcwd()+"/../Data/"+sys.argv[testId + 1]+"/"
	FScore = open(Tpath+sys.argv[trainId + 4], 'r+').read().splitlines()
	Tools  = FScore[0].split(":")[1].split()
	##-------- Accumulate the data from all the tools -------
	[xdata,ydata] = func.genAllTestData(Tpath+"Training/",Tools)
	for tool in Tools:
		TestToolDict[tool] = func.getPredictedScoreError(xdata,ydata,w[tool])
	func.reportTrend(TestToolDict, "Testing on SVComp15")

    ##------------ Here we do category-wise testing -----------------------##
	##--- Category specific test vector -----
	Tpath     = os.getcwd()+"/../Data/"+sys.argv[testId + 1]+"/"	
	TBMSubDict = func.getBMSubDict(open(Tpath+sys.argv[testId + 3], 'r+').read().splitlines())
	print TBMSubDict
	ToolCategoryScore = dict(dict([]))

	for c in TBMSubDict.keys():
		cxdata = []
		cydata = []
		cxdata = func.getCategoryData(genDataTest,c,TBMSubDict[c] )
		tdict = dict([])
		for t in Tools:
			cydata = [scoreTest[t][c]]*len(cxdata)
			tdict[t] = func.getPredictedScoreError2(cxdata,cydata,w[t])
			#print "Break:", ToolCategoryScore[c]
		ToolCategoryScore[c] = tdict
		print "Break:",c, [tdict[i][1] for i in Tools]
		func.reportTrend(ToolCategoryScore[c]," prediction For Category-"+c+" on svcomp15 ")
		


#	CategoryWinner = getPerCategoryWinner(ToolCategoryScore)
			
		
			
		
	
	
	
   
	
			
