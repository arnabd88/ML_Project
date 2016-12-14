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
        

print "Generating Training and Test Data ..............................."
[[genDataTrain,scoreTrain], [genDataTest,scoreTest]] = func.genData(trainId, testId)
print "Completed Generating Training and Test Data ...............................\n"

#################################################################################

##: LINEAR REGRESSION 

#################################################################################

##------ Train on every tool -------##

w = {}
epoch = 5
if(learnId != -1):
	#print "Initiating Linear Regression learning .....................\n"
        Dpath  = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
      	FScore = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()
        Tools  = FScore[0].split(":")[1].split()

        TrainToolsDict = {}
	testDataDict = {}
        for tool in Tools:
	     accuracy_lin = {}
	     w_temp2 = {}
	     TrainToolsDict_temp = {}
             accuracy_lst = []
	     ##--------- Get the training data ---------##
             dataFileName = Dpath+"/Training/"+tool+".data"
	     allData = open(dataFileName, "rb").read().splitlines()

	     for t in range(epoch):
		     random.shuffle(allData)
		     trainLength = int(len(allData) * 0.6)
        	     testLength = len(allData) - trainLength
	             trainData = []
        	     for i in range(trainLength):
                	trainData.append(random.choice(allData))

		     [xTrainData, yTrainData] = func.parseInfo(trainData)
	             w_temp2[t] = linearRegression.LinearRegression(xTrainData, yTrainData)
		     
		     ##---- Get the error norm and score on the training data -----##
	             TrainToolsDict_temp[t] = func.getPredictedScoreError(xTrainData, yTrainData, w_temp2[t])
        	     testData = [item for item in allData if item not in trainData]
		     [xTestData, yTestData] = func.parseInfo(testData)
		     accuracy_lst.append(func.accuracy(w_temp2[t], xTestData, yTestData))

	     #func.reportAccuracy(tool, accuracy_lin[tool])
	     
	     ## --- Get the mean of acuuracies --- ##
	     accuracy_lin[tool] = sum(accuracy_lst)/len(accuracy_lst) * 100
 	     func.reportAccuracy(tool, accuracy_lin[tool])
	    
	     ## --- Get the mean of weights --- ##
	     w_temp2_array = [w_temp2[i] for i in w_temp2.keys()]
	     w_temp2_t = numpy.transpose(w_temp2_array)
	     w[tool] = [sum(w_temp2_t[i])/len(w_temp2_t[i]) for i in range(0, len(w_temp2_t))]
	
	     # --- Get the mean of errors and scores --- ##
	     TrainToolsDict_temp_array = [TrainToolsDict_temp[i] for i in TrainToolsDict_temp.keys()]
	     TrainToolsDict_temp_t = numpy.transpose(numpy.array(TrainToolsDict_temp_array))
	     TrainToolsDict[tool] = [[], sum(TrainToolsDict_temp_t[1])/len(TrainToolsDict_temp_t[1])]



 
	     ##---- Get the error norm and score on the training data -----##
             #TrainToolsDict[tool] = func.getPredictedScoreError(xTrainData, yTrainData, w_temp)
	     #testDataDict[tool] = [item for item in allData if item not in trainData]
	     #print "Completed Linear Regression learning .....................\n"

        func.reportTrend(TrainToolsDict, "Linear: Training on SVComp14")

##------ Testing on SVComp15 (Predict the overall Winner)----------
if(testId!=-1 and trainId!=-1): ##indicates training dataStructure is available
	
	TestToolDict = {}
	##########################################################
	
	##: OVERALL TESTING ON TEST DATASET

	##########################################################

	##-------- Accumulate the data from all the tools -------
	print "Overall testing on test dataset .....................\n"

	for tool in Tools:
		dataFileName = Dpath+"/Training/"+tool+".data"
  	        testData = open(dataFileName, "rb").read().splitlines()
		[xTestData,yTestData] = func.parseInfo(testData)
		TestToolDict[tool] = func.getPredictedScoreError(xTestData,yTestData,w[tool])
	func.reportTrend(TestToolDict, "Linear: Overall Testing on Test Dataset")
	print "Completed Overall Testing on Test Dataset .....................\n"

		
#################################################################################

##: LINEAR REGRESSION with NON-LINEAR TRANSFORMATIONS AND REGULARIZATION

#################################################################################
##---- Using non-linear transforms upto 3rd order ---
##-- \phi(x) = [1,x,.5(3x^2-1), .5(5x^3 - 3x)]	

w_nonLin = {}
lam = [0.0001, 0.01, 0.1, 1, 10]
	
if(learnId != -1):
	print "Initiating NonLinear Regression learning .....................\n"
	#Dpath  = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
	#FScore = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()
	#trainTools  = FScore[0].split(":")[1].split()
	
	#w = {} #[tool -> weight([index -> val])]
	TrainToolsDict = {}
	testDataDict = {}
	accuracy = {}
	for tool in Tools:
	     w_temp2 = {}
             TrainToolsDict_temp = {}
	     accuracy_temp = []
     	     ##--------- Get the training data ---------##
  	     dataFileName = Dpath+"/Training/"+tool+".data"
             allData = open(dataFileName, "rb").read().splitlines()

	     for t in range(epoch):
                     random.shuffle(allData)
                     trainLength = int(len(allData) * 0.6)
                     testLength = len(allData) - trainLength
                     trainData = []
                     for i in range(trainLength):
                        trainData.append(random.choice(allData))

                     [xTrainData, yTrainData] = func.parseInfo(trainData)
		     Nxdata = func.nonLinTransform(xTrainData)
        	     bestLam, bestAcc = func.kFoldCV(10, Nxdata, yTrainData, lam)
	             #print "Best lam: " + str(bestLam) + ", best Accuracy:" + str(bestAcc)
		     	
                     w_temp2[t] = func.linReg(Nxdata,yTrainData, bestLam)
		     
		     TrainToolsDict_temp[t] = func.getPredictedScoreError(Nxdata,yTrainData,w_temp2[t])

		     testData = [item for item in allData if item not in trainData]
	
                     [xTestData, yTestData] = func.parseInfo(testData)
        	     Nxdata = func.nonLinTransform(xTestData)
		     accuracy_temp.append(func.accuracy(w_temp2[t], Nxdata, yTestData))	

	     
	     ## --- Get the mean of acuuracies --- ##                                         
             accuracy[tool] = sum(accuracy_temp)/len(accuracy_temp) * 100
	     #print accuracy[tool]
	     func.reportAccuracy(tool, accuracy[tool])

	     ## --- Get the mean of weights --- ##
             w_temp2_array = [w_temp2[i] for i in w_temp2.keys()]
             w_temp2_t = numpy.transpose(w_temp2_array)
             w_nonLin[tool] = [sum(w_temp2_t[i])/len(w_temp2_t[i]) for i in range(0, len(w_temp2_t))]

	     # --- Get the mean of errors and scores --- ##
             TrainToolsDict_temp_array = [TrainToolsDict_temp[i] for i in TrainToolsDict_temp.keys()]
             TrainToolsDict_temp_t = numpy.transpose(numpy.array(TrainToolsDict_temp_array))
             TrainToolsDict[tool] = [[], sum(TrainToolsDict_temp_t[1])/len(TrainToolsDict_temp_t[1])]

	print "Completed NonLinear Regression learning .....................\n"
	
	func.reportTrend(TrainToolsDict, "NonLinear: Training on SVComp14")


##------ Testing on test dataset (Predict the overall Winner)----------
if(testId!=-1 and trainId!=-1): ##indicates training dataStructure is available
	
	TestToolDict = {}
	#Tpath = os.getcwd()+"/../Data/"+sys.argv[testId + 1]+"/"
	#FScore = open(Tpath+sys.argv[trainId + 4], 'r+').read().splitlines()
	#Tools  = FScore[0].split(":")[1].split()
        #commonTools = [item for item in Tools if item in trainTools]

	##########################################################
	
	##: TESTING ON FULL SVCOMP15 DATASET

	##########################################################

	print "Overall Testin test dataset .....................\n"
	for tool in Tools:
		dataFileName = Dpath+"/Training/"+tool+".data"
		testData = open(dataFileName, "rb").read().splitlines()
                [xTestData, yTestData] = func.parseInfo(testData)
                Nxdata = func.nonLinTransform(xTestData)		
		TestToolDict[tool] = func.getPredictedScoreError(Nxdata,yTestData,w_nonLin[tool])
	print "Completed Overall Testing on test dataset .....................\n"
	func.reportTrend(TestToolDict, "NonLinear: Testing on Test Dataset")

