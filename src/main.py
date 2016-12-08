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
        
if(trainId != -1):
	Dpath     = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
	print Dpath+sys.argv[trainId + 2]
	BMList    = open(Dpath+sys.argv[trainId + 2], 'r+').read().splitlines()
	BMSubList = open(Dpath+sys.argv[trainId + 3], 'r+').read().splitlines()
	FScore    = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()

else:
	##--- Add code for using learned weights ---
	print "Use Learned weights"

if(genId != -1):
        func.generateTrainingData(BMList, BMSubList, FScore, Dpath)

##------ Train on every tool -------##
if(learnId != -1):
        Tools = []
	
        if 'FScore' in locals():
                Tools = FScore[0].split(":")[1].split()
        else:
                Dpath  = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
      	        FScore = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()
                Tools  = FScore[0].split(":")[1].split()

        w = {} #[tool -> weight([index -> val])]
        for tool in Tools:
                ##--------- Get the training data ---------##
                trainFileName = Dpath+"/Training/"+tool+".data"

                trainData = csv.reader(open(trainFileName, "rb"), delimiter=" ")

                w_temp = linearRegression.LinearRegression(trainData)
                w[tool] = w_temp

        print "w:" , w
