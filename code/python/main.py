'''
 * @date  : 2020-11-17
 * @author: Rohit S <rohit2013107@iitgoa.ac.in>
 * @title : Gaussian Naive Bayes Implementation
 * Driver program to test the NaiveBayes class
 * against an EMNIST dataset of 370,000+ grayscale 
 * 28x28 images of A-Z handwritten alphabets
'''

import sys                                              # read cmd line input
import numpy as np                                      # faster mathematical operations
import seaborn as sns                                   # display confusion matrix as heatmap
import matplotlib.pyplot as plt                         # display mean and variance as images
from progress.bar import IncrementalBar                 # display progressbar during processing
from NaiveBayes import NaiveBayes                       # my NaiveBayes class

# get dataset path from command line input
if len(sys.argv) == 1:                                  # if cmd line arg not provided
	print("provide file path as cmd line argument")     # show error
	sys.exit()                                          # force exit

# setup & init
# ------------
classes    = 26                                         # number of classes
feautures  = 28*28                                      # number of pixels
smoothen   = 2048                                       # smoothing factor
label      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"               # human-friendly class labels
classifier = NaiveBayes(classes, feautures, smoothen)   # my NaiveBayes classifier
testQ      = []                                         # queue of samples to be tested
testN      = np.zeros(classes, np.uint)                 # tests performed per class
confusion  = np.zeros((classes, classes), np.uint)      # confusion matrix

# read file
# ---------
bar = IncrementalBar("Training: ", max = 372451)        # @ display progress bar 
with open(sys.argv[1]) as infile:                       # dataset path from command line input
	for line in infile:                                 # for each line
		sample = np.array(line.split(','), np.float)    # 1+(28*28) elements, read as float
		if np.random.randint(0,9) > 0:                  # 90% samples (randomly selected) for training
			classifier.train(int(sample[0]),sample[1:]) # -- first element is label, rest are features
		else:                                           # remaining 10% added to testing queue
			testQ.append(sample)                        # -- add sample to testQ
		bar.next()                                      # @ increment progress bar
bar.finish()                                            # @ close progress bar

# perform tests
# -------------
bar = IncrementalBar("Testing:  ", max = len(testQ))    # @ display progress bar 
for sample in testQ:                                    # for each sample to be tested
	actual    = int(sample[0])                          # actual label
	predicted = classifier.predict( sample[1:] )        # predicted label
	confusion[actual][predicted] += 1                   # add to confusion matrix
	testN[actual] += 1                                  # add to row-wise sum
	bar.next()                                          # @ increment progress bar
bar.finish()                                            # @ close progress bar

# accuracy
# --------
total = 0
successes = 0
print("Accuracy per class:")                            # accuracy per class
for i in range(classes):                                # for each class
	successes += confusion[i][i];                       # -- total successes
	total     += testN[i];                              # -- total tests
	accuracy   = confusion[i][i]/testN[i];              # -- accuracy = successes/tests
	print("%c %.2f%%" % (label[i], accuracy * 100) )    # -- print as percent with proper label
accuracy = successes / total                            # total accuracy
print("Overall accuracy: %.2f%%" % (accuracy*100) )     # print overall accuracy as percent

# mean & variance
# ---------------
for arr in [classifier.mean, classifier.varn]:          # for both, mean and augmented variance
	plt.figure()                                        # create new figure
	for i in range(classes):                            # for each class
		img = arr[i].reshape(28,28)                     # convert array to 28x28 matrix
		plt.subplot(3, 9, i+1)                          # set up a 3x9 grid for 26 letters
		plt.title(label[i])                             # print label of item being displayed
		plt.axis("off")                                 # x and y axes not required
		plt.imshow(img,                                 # display as an image
			cmap = "gray",                              # -- grayscale (no color)
			interpolation = "quadric")                  # -- smoothen edges of image

# confusion matrix
# ----------------
plt.figure()                                            # create new figure
ax = sns.heatmap(confusion,                             # display confusion matrix as heatmap
		cmap="Blues",                                   # heatmap color scheme
		annot=True,                                     # show the values long with the colors
		fmt='g',                                        # disable scientific notations for values
		xticklabels = list(label),                      # use human-readable column-labels
		yticklabels = list(label) )                     # use human-readable row-labels
plt.xlabel('Predicted class value')                     # label for x-axis 
plt.ylabel('True class value')                          # label for y-axis

# display
# -------
plt.show()