
import sys
import re
import copy
import func
import math
import numpy
import os

trainId = -1

if("-trainDir" in sys.argv):
	trainId = sys.argv.index("-trainDir")
else:
	print "No Training. Use already trained weights"

if(trainId != -1):
	Dpath        = os.getcwd()+"/../Data/"+sys.argv[trainId + 1]+"/"
	print Dpath+sys.argv[trainId + 2]
	BMList    = open(Dpath+sys.argv[trainId + 2], 'r+').read().splitlines()
	BMSubList = open(Dpath+sys.argv[trainId + 3], 'r+').read().splitlines()
	FScore    = open(Dpath+sys.argv[trainId + 4], 'r+').read().splitlines()
else:
	##--- Add code for using learned weights ---
	print "Use Learned weights"

func.generateTrainingData(BMList, BMSubList, FScore, Dpath)


