import evaluation_SRL as E
import sys
from os import walk

def getPR(gold, test):
        eval = E.Eval(gold, test)
        precision = eval.getTotalPrecision()
        recall = eval.getTotalRecall()
        F = 2*precision*recall/(precision + recall)
        return precision, recall, F

def printPR(gold, test):
	precision, recall, F = getPR(gold, test)
	print "precision : ", precision, "\trecall : ", recall, "\tF : ", F

def getFiles(predDir):
	files = []
        for (dirpath, dirnames, filenames) in walk(predDir):
                files.extend(filenames)
                break
        #files.sort(key = str.split(".")[0].split("_")[-1])
	files.sort(key = lambda s: int(s.split(".")[0].split("_")[-1]))
	return files
	

def getAllPR(goldD, goldT, predDir):
	predDevDir = "predDevFiles/"
	predTestDir = "predTestFiles/"

	if(predDir[-1] != "/"):
		predDir += "/"

	predDevDir = predDir + predDevDir
	predTestDir = predDir + predTestDir

	print predDevDir, predTestDir	

	devFiles = getFiles(predDevDir)
	testFiles = getFiles(predTestDir)

	 
	
	if(len(devFiles) != len(testFiles)):
		sys.exit()

	for i in range(0, len(devFiles)):
		epoch = devFiles[i].split(".")[0].split("_")[-1]
		print "For epoch : ", epoch
		testD = predDevDir + devFiles[i]
		testT = predTestDir + testFiles[i]
		print "Devl : ",
		printPR(goldD, testD)
		print "Test : ",
		printPR(goldT, testT)


if __name__ == "__main__":
	goldD = sys.argv[1]
	goldT = sys.argv[2]
	
	#test = sys.argv[2]
	predDir = sys.argv[3]
	#printPR(gold, test)
	getAllPR(goldD, goldT, predDir)

