import os.path
import sys
from collections import defaultdict
from operator import itemgetter

# A class for evaluating POS-tagged data
class Eval:
    def __init__(self, goldFile, testFile):
        print "Evaluation Script for computing precision and recall"
        self.goldTags = self.readLabeledData(goldFile)
        self.testTags = self.readLabeledData(testFile)
        self.tags = self.learnTags()
        
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = set();
            sens_id=0
            for line in file:
                start = 0
                raw = line.split()
                if(len(raw)>0):
                    tags = []
                    for token in raw:
                        parts = token.split('_');
                        tags.append(parts[1]) # list of tags in the sentence
                    while(start != len(tags)):
                        end = start
                        for end in range(start,len(tags)):
                            if(end != len(tags)-1): #prevent out of range index
                                if(tags[end+1] != tags[end]):
                                    break
                        if(tags[start]!='PRED'):
                            phrase = (sens_id, start, end, tags[start])
                        start = end + 1                            
                        sens.add(phrase)
                sens_id += 1               
            return sens
        else:
            print "Error: unlabeled data file", inputFile, "does not exist"  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
            
    
    def learnTags(self):
        tags = set()
        for sen in self.goldTags:
            temp = list(sen)
            tags.add(temp[3])
        return list(tags)
            
                    
    def getTotalPrecision(self):        
        total = len(self.testTags)
        correct = len( self.goldTags & self.testTags)
        return float(correct)/float(total)
        
    def getTotalRecall(self):        
        total = len(self.goldTags)
        correct = len( self.goldTags & self.testTags)
        return float(correct)/float(total)
 
    def getPrecision(self,tagTi):
        testset = set()
        goldset = set()
        for tag in self.testTags:
            temp = list(tag)
            if(temp[3] == tagTi):
                testset.add(tag)
        for tag in self.goldTags:
            temp = list(tag)
            if(temp[3] == tagTi):
                goldset.add(tag)
        total = len(testset)
        correct = len(testset & goldset)
        toret = float(correct)/float(total)
        #print "\t Precision when predicting argument "+tagTi+" is "+str(toret)
        return toret   
         

    def getRecall(self,tagTi):
        testset = set()
        goldset = set()
        for tag in self.testTags:
            temp = list(tag)
            if(temp[3] == tagTi):
                testset.add(tag)
        for tag in self.goldTags:
            temp = list(tag)
            if(temp[3] == tagTi):
                goldset.add(tag)
        total = len(goldset)
        correct = len(testset & goldset)
        toret = float(correct)/float(total)
        #print "\t Recall when predicting argument "+tagTi+" is "+str(toret)
        return toret
        

def colform(txt, width):
    if len(txt) > width:
        txt = txt[:width]
    elif len(txt) < width:
        txt = txt + (" " * (width - len(txt)))
    return txt           
                                 
if __name__ == "__main__":
    # Pass in the gold and test labelled data as arguments
    if len(sys.argv) < 2:
        print "Call evaluation_SRL.py with two arguments: gold.txt and test.txt"
    else:
#        outFile = 'C:\Users\srihita\Desktop\Rplots\Output.txt'

        gold = sys.argv[1]
        test = sys.argv[2]
        eval = Eval(gold, test)
        # Calculate recall and precision
        line = "Arguments" + "\t" + "\t" +"Precision" + "    " + "Recall" + "\t     " + "F"
        print >>file, line.rstrip()  
        for tag in eval.tags:
            precision = eval.getPrecision(tag)
            recall = eval.getRecall(tag)
            F = 2*precision*recall/(precision + recall)
            line = colform(tag,7) + "\t" + "\t" +"\t" + str(round(precision,4)) + "\t     " + str(round(recall,4)) + "\t     " + str(round(F,4))
            print >>file, line.rstrip()
        precision = eval.getTotalPrecision()
        recall = eval.getTotalRecall()
        F = 2*precision*recall/(precision + recall)
        line = colform("total",7) + "\t" + "\t" +"\t" + str(round(precision,4)) + "\t     " + str(round(recall,4)) + "\t     " + str(round(F,4))
        print >>file, line.rstrip()
     
            
