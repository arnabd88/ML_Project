import csv
import sys
import re
import copy
import math
import numpy
#import matplotlib.pyplot as plt
import os
import random

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
print "Generating Training and Test Data ..............................."
[[genDataTrain,scoreTrain], [genDataTest,scoreTest]] = func.genData(trainId, testId)
print "Completed Generating Training and Test Data ...............................\n"

#################################################################################

##: LINEAR REGRESSION 

#################################################################################

##------ Train on every tool -------##

w = {}
if(learnId != -1):
        Tools = []
	
	    #print "Initiating Linear Regression learning .....................\n"
        Dpath  = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
      	FScore = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()
        trainTools  = FScore[0].split(":")[1].split()

        TrainToolsDict = {}
        for tool in trainTools:
             ##--------- Get the training data ---------##
             trainFileName = Dpath+"/Training/"+tool+".data"

             #trainData = csv.reader(open(trainFileName, "rb"), delimiter=" ")
	     
	     trainData = open(trainFileName, "rb").read().splitlines()	     
	     
	     trainLength = int(len(trainData) * 0.6)
	     testLength = len(trainData) - trainLength
	     trainData2 = []
	     for i in range(trainLength):
		trainData2.append(random.choice(trainData))
	
	     [xdata,ydata] = func.parseInfo(trainData2)
             w_temp = linearRegression.LinearRegression(xdata, ydata)
             w[tool] = w_temp

			 ##---- Get the error norm and score on the training data -----##
             TrainToolsDict[tool] = func.getPredictedScoreError(xdata,ydata, w_temp)
		#print "Completed Linear Regression learning .....................\n"

        func.reportTrend(TrainToolsDict, "Linear: Training on SVComp14")

     #   print "w:" , w

##------ Testing on SVComp15 (Predict the overall Winner)----------
if(testId!=-1 and trainId!=-1): ##indicates training dataStructure is available
	
	TestToolDict = {}
	Tpath = os.getcwd()+"/../Data/"+sys.argv[testId + 1]+"/"
	FScore = open(Tpath+sys.argv[trainId + 4], 'r+').read().splitlines()
	Tools  = FScore[0].split(":")[1].split()
	commonTools = [item for item in Tools if item in trainTools]

	##########################################################
	
	##: TESTING ON FULL SVCOMP15 DATASET

	##########################################################

	##-------- Accumulate the data from all the tools -------
	print "Testing on Full SVComp15 dataset .....................\n"

	#[xdata,ydata] = func.genAllTestData(Tpath+"Training/",Tools)
	[xdata,ydata] = func.genAllTestData(Tpath+"Training/",commonTools)
	for tool in commonTools: #Tools
		TestToolDict[tool] = func.getPredictedScoreError(xdata,ydata,w[tool])
	func.reportTrend(TestToolDict, "Linear: Testing on SVComp15")
	print "Completed Testing on Full SVComp15 dataset .....................\n"

    ##------------ Here we do category-wise testing -----------------------##
	##########################################################
	
	##: CATEGORY SPECIFIC TESTING ON SVCOMP15 DATASET

	##########################################################

	##--- Category specific test vector -----
	Tpath     = os.getcwd()+"/../Data/"+sys.argv[testId + 1]+"/"	
	TBMSubDict = func.getBMSubDict(open(Tpath+sys.argv[testId + 3], 'r+').read().splitlines())
	#print TBMSubDict
	ToolCategoryScore = dict(dict([]))

	print "=============== Linear: Category Specific Testing on SVComp15 ======================\n"
	print "Testing Linear: Category Specific Testing on SVComp15 ======================\n"
	for c in TBMSubDict.keys():
		cxdata = []
		cydata = []
		cxdata = func.getCategoryData(genDataTest,c,TBMSubDict[c] )
		tdict = dict([])
		for t in commonTools:
			cydata = [scoreTest[t][c]]*len(cxdata)
			tdict[t] = func.getPredictedScoreError2(cxdata,cydata,w[t])
			#print "Break:", ToolCategoryScore[c]
		ToolCategoryScore[c] = tdict
		#print "Break:",c, [tdict[i][1] for i in Tools]
		func.reportTrend(ToolCategoryScore[c]," prediction For Category-"+c+" on svcomp15 ")
	print "Completed Testing Linear: Category Specific Testing on SVComp15 ======================\n"
		

#################################################################################

##: LINEAR REGRESSION with NON-LINEAR TRANSFORMATIONS AND REGULARIZATION

#################################################################################

			
	
##---- Using non-linear transforms upto 3rd order ---
##-- \phi(x) = [1,x,.5(3x^2-1), .5(5x^3 - 3x)]	

w_nonLin = {}
lam = [0.1] #[0.0001, 0.01, 0.1, 1, 10]
	
if(learnId != -1):
	print "Initiating NonLinear Regression learning .....................\n"
	Dpath  = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
	FScore = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()
	trainTools  = FScore[0].split(":")[1].split()
	
	#w = {} #[tool -> weight([index -> val])]
	TrainToolsDict = {}
	for tool in trainTools:
		##--------- Get the training data ---------##
		trainData = open(Dpath+"/Training/"+tool+".data",'r+').read().splitlines()
		[xdata,ydata] = func.parseInfo(trainData) ## raw data
		Nxdata = func.nonLinTransform(xdata)
		bestLam, bestAcc = func.kFoldCV(25, Nxdata, ydata, lam)
		print "Best lam: " + str(bestLam) + ", best Accuracy:" + str(bestAcc)
		w_nonLin[tool] = func.linReg(Nxdata,ydata, bestLam)
		
		TrainToolsDict[tool] = func.getPredictedScoreError(Nxdata,ydata, w_nonLin[tool])
	print "Completed NonLinear Regression learning .....................\n"
	
	func.reportTrend(TrainToolsDict, "NonLinear: Training on SVComp14")

##------ Testing on SVComp15 (Predict the overall Winner)----------
if(testId!=-1 and trainId!=-1): ##indicates training dataStructure is available
	
	TestToolDict = {}
	Tpath = os.getcwd()+"/../Data/"+sys.argv[testId + 1]+"/"
	FScore = open(Tpath+sys.argv[trainId + 4], 'r+').read().splitlines()
	Tools  = FScore[0].split(":")[1].split()
        commonTools = [item for item in Tools if item in trainTools]

	##########################################################
	
	##: TESTING ON FULL SVCOMP15 DATASET

	##########################################################

	print "Testing on Full SVComp15 dataset .....................\n"
	##-------- Accumulate the data from all the tools -------
	[xdata,ydata] = func.genAllTestData(Tpath+"Training/",Tools)
	Nxdata = func.nonLinTransform(xdata)
	for tool in commonTools:
		TestToolDict[tool] = func.getPredictedScoreError(Nxdata,ydata,w_nonLin[tool])
	print "Completed Testing on Full SVComp15 dataset .....................\n"
	func.reportTrend(TestToolDict, "NonLinear: Testing on SVComp15")

    ##------------ Here we do category-wise testing -----------------------##
	##########################################################
	
	##: CATEGORY SPECIFIC TESTING ON SVCOMP15 DATASET

	##########################################################


	##--- Category specific test vector -----
	Tpath     = os.getcwd()+"/../Data/"+sys.argv[testId + 1]+"/"	
	TBMSubDict = func.getBMSubDict(open(Tpath+sys.argv[testId + 3], 'r+').read().splitlines())
	#print TBMSubDict
	ToolCategoryScore = dict(dict([]))

	print "Testing NonLinear: Category Specific Testing on SVComp15 ======================\n"
	for c in TBMSubDict.keys():
		cxdata = []
		cydata = []
		cxdata = func.getCategoryData(genDataTest,c,TBMSubDict[c] )
		Nxdata = func.nonLinTransform(cxdata)
		#print len(Nxdata)
		tdict = dict([])
		for t in commonTools:
			cydata = [scoreTest[t][c]]*len(Nxdata)
			tdict[t] = func.getPredictedScoreError2(Nxdata,cydata,w_nonLin[t])
			#print "Break:", ToolCategoryScore[c]
		ToolCategoryScore[c] = tdict
		#print "Break:",c, [tdict[i][1] for i in Tools]
		func.reportTrend(ToolCategoryScore[c]," prediction For Category-"+c+" on svcomp15 ")
	print "Completed Testing NonLinear: Category Specific Testing on SVComp15 ======================\n"
	
	#errors = []
	#for i in range(len(TrainToolsDict.values())):
	#    errors.append(TrainToolsDict.values()[0])

    	#plt.plot(numpy.array(TrainToolsDict.keys()), numpy.array(errors), 'bs')
    	#plt.xlabel('Tools')
	#plt.ylabel('Prediction error')
   	#plt.title('Prediction error for each tool')
	#plt.show()
		

	
	
   
	
			
