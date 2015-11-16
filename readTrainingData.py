import os

class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]


def readLabeledData(inputFile):
    if os.path.isfile(inputFile):
        file = open(inputFile, "r") 
        sens = [];
        
        # iterate over lines
        for line in file:
            raw = line.split()
            sentence = []
            
            # iterate over tokens
            for token in raw:
                sentence.append(TaggedWord(token))
            
            sens.append(sentence) 

        file.close()
    else:
        print('File does not exist')
    return sens

    
def createVocab(sens, freq_thresh):
    vocab = dict()
    freq = dict()
    filtered_freq = dict()
    inv_vocab = dict()
    labels = dict()

    labelCount = 0
    for sentence in sens:
        for taggedWord in sentence:
            word  = taggedWord.word
            label = taggedWord.tag

            if word in freq:
                freq[word] = freq[word] + 1
            else:
                freq[word] = 1
                
            if label not in labels:
                labels[label] = labelCount
                labelCount = labelCount + 1
            
    # remove words with frequency less than a thresh
    count_unk = 0
    for word, word_freq in freq.iteritems():
        if word_freq > freq_thresh:
            filtered_freq[word] = word_freq
        else:
            count_unk = count_unk + 1

    # add remaining words to vocab
    count = 0
    for word in filtered_freq:
        vocab[word] = count
        inv_vocab[count] = word
        count = count + 1

    # add unk
    vocab['unk'] = count
    inv_vocab[count] = 'unk'
    filtered_freq['unk'] = count_unk

    return vocab, inv_vocab, filtered_freq, labels    


def oneHotFeatVec(word,vocab):
    vocabLen = len(vocab)
    
    if word in vocab:
        return vocab[word]
    else:
        return vocabLen-1

class featSequencer():
    def __init__(self, sentence, vocab, labels):
        featSeq = []
        labelSeq = []
        for taggedWord in sentence:
            labelSeq.append(labels[taggedWord.tag])
            featSeq.append(oneHotFeatVec(taggedWord.word, vocab))
        self.featSeq = featSeq
        self.labelSeq = labelSeq

    def writeToFile(self, file):
        file.write('{\n')
        for i in range(0,len(self.featSeq)):
            file.write(str(self.featSeq[i]) + ' ' + str(self.labelSeq[i]) + '\n')
        file.write('}\n')
        
if __name__=="__main__":
    inputFile = '/home/tgupta6/pos_tag_data/train.txt'
    parsedSensFile = '/home/tgupta6/pos_tag_data/sentences.txt'
    vocabularyFile = '/home/tgupta6/pos_tag_data/vocab.txt'
    labelsFile = '/home/tgupta6/pos_tag_data/labels.txt'
    featureFile = '/home/tgupta6/pos_tag_data/feat.txt'

    FREQ_THRESH = 5

    # parse training data
    sens = readLabeledData(inputFile)

    # write sentences to a file
    sensFile = open(parsedSensFile,'w')
    for sentence in sens:
        for taggedWord in sentence:
            sensFile.write(taggedWord.word + " ")
        sensFile.write('\n')
    sensFile.close()
    
    # Create vocabulary
    vocab, inv_vocab, freq, labels = createVocab(sens, FREQ_THRESH)

    # write vocabulary to a file
    vocabFile = open(vocabularyFile,'w')
    for idx, word in inv_vocab.iteritems():
        word_freq = freq[word]
        vocabFile.write(str(idx) + ' ' + word + ' ' + str(word_freq) + '\n')
    vocabFile.close()

    # write labels to a file
    labFile = open(labelsFile,'w')
    for label in labels:
        labFile.write(label + ' ' + str(labels[label]) + '\n' )
    labFile.close()

    # write one hot feature vectors to file
    featFile = open(featureFile, 'w')
    sentence_count = 0
    for sentence in sens:
#        print(sentence_count)
        sentence_count = sentence_count + 1
        F = featSequencer(sentence,vocab,labels)
        F.writeToFile(featFile)
    featFile.close()


