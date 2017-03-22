import numpy as np
import operator

def createDataSet():
	group = np.array([[1, 1], [1, 1.2], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	distance = sqDiffMat.sum(axis=1)
	sortedDistIndicies = distance.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fileObj = open(filename, 'r')
	lines = fileObj.readlines()
	numOfInput = len(lines[0].split('\t')) - 1
	numOfLines = len(lines)
	mat = np.zeros((numOfLines, numOfInput))
	label = list()
	for i in range(numOfLines):
		line  = lines[i].strip()
		listFrLine = line.split('\t')
		mat[i, :] = listFrLine[0:numOfInput]
		label.append(int(listFrLine[-1]))
	return mat, label

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVals, (m,1))
	normDataSet = normDataSet / np.tile(ranges, (m,1))
	return normDataSet, minVals, ranges

def dataTest(dataFile = 'datingTestSet.txt', k = 3, ratio = 0.2):
	mat, label = file2matrix(dataFile)
	normMat, minVals, ranges = autoNorm(mat)
	dataSize = normMat.shape[0]
	testSize = int(ratio * dataSize)
	errorCount = 0
	for i in range(testSize):
		result = classify0(normMat[i, :], normMat[testSize:dataSize, :], label[testSize:dataSize], k)
		print '%d. classifier answer is %s, real answer is %s' % (i, result, label[i])
		if result != label[i] :
			errorCount += 1
	print 'classifier error rate is %f' % (float(errorCount) / float(testSize))

if __name__ == '__main__':
	dataTest()
