import sys
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar as ProgressBar

if len(sys.argv) == 1:
	print("provide file path as cmd line argument")
	sys.exit()

random.seed()
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def NaiveBayes(dataset, testset, smoothen):
	# convert to np array for operations
	dataset = np.array(dataset)
	testset = np.array(testset)

	# ----- training -----
	bar = ProgressBar("Training classifier ", max=4)
	# label and frequency of each class in dataset
	label, freq = np.unique(dataset[:,0], return_counts=True)
	bar.next()
	# mean of values per attribute per class
	mean = [np.mean(dataset[dataset[:,0]==i, 1:], axis=0) for i in label]
	bar.next()
	# variance of values per attribute per class + smoothing factor
	varn = [np.var(dataset[dataset[:,0]==i, 1:], axis=0, ddof=1)*2 + smoothen for i in label]
	bar.next()
	# constant term
	m = [(-np.log(freq[k]) + np.sum(np.log(varn[k]))/2) for k in range(len(label))]
	bar.next()
	bar.finish()

	# ----- testing -----
	bar = ProgressBar("Testing classifier  ", max=testset.shape[0])
	confusion = np.zeros([len(label),len(label)])
	for i in range(testset.shape[0]):
		# predicted class label (see equation in notes)
		predict = np.argmin([
			m[j] + np.sum( ((testset[i,1:]-mean[j])**2)/varn[j] )
			for j in range(len(label))
		])
		# actual class label
		actual  = int(testset[i,0]) 
		# confusion matrix
		confusion[actual][predict] += 1
		bar.next()
	bar.finish()

	# ----- return -----
	return(confusion, mean, varn)


# read data set
# -------------
dataset = []
testset = []
bar = ProgressBar("Reading data samples", max=372451)
with open(sys.argv[1]) as infile:
	for line in infile:
		sample = [int(x) for x in line.split(',')]
		if np.random.randint(0,9) == 0:
			testset.append(sample)
		else:
			dataset.append(sample)
		bar.next()
bar.finish()

(confusion, mean, variance) = NaiveBayes(dataset, testset, 32*32)


# print report
# ------------
bar = ProgressBar("Saving results      ", max=3)

# mean of the attributes of each class, as 28*28 image
plt.figure(figsize=(18,6))
for i in range(len(mean)):
	img = mean[i].reshape(28,28)
	plt.subplot(3, 9, i+1)
	plt.title(alphabet[i])
	plt.axis("off")
	plt.imshow(img, cmap = "gray", interpolation = "quadric")
plt.subplots_adjust(hspace=0.25)
plt.savefig('output/mean.png', bbox_inches = 'tight')
bar.next()

# variance of the attributes of each class, as 28*28 image
plt.figure(figsize=(18,6))
for i in range(len(variance)):
	img = variance[i].reshape(28,28)
	plt.subplot(3, 9, i+1)
	plt.title(alphabet[i])
	plt.axis("off")
	plt.imshow(img, cmap = "gray", interpolation = "quadric")
plt.subplots_adjust(hspace=0.25)
plt.savefig('output/variance.png', bbox_inches = 'tight')
bar.next()

# confusion matrix as heatmap
plt.figure(figsize=(16,9))
ax = sns.heatmap(confusion, annot=True, fmt='g', cmap="Blues", 
					xticklabels = list(alphabet), yticklabels = list(alphabet) )
plt.xlabel('Predicted class value')
plt.ylabel('True class value')
plt.savefig('output/confusion.png', bbox_inches = 'tight')
bar.next()

bar.finish()

# accuracy
accuracy = np.trace(confusion) / np.sum(confusion)
print("Overall accuracy: %.2f%%" % (accuracy*100) )
print("Accuracy per class:")
for i in range(confusion.shape[0]): 
	accuracy = confusion[i][i] / np.sum(confusion[i])
	print("%c %8.2f%%" % (alphabet[i], accuracy * 100) )