'''
@date  : 2020-11-17
@author: Rohit S <rohit2013107@iitgoa.ac.in>
@title : Gaussian Naive Bayes Implementation
Gaussian Naive Bayes classifier that can work with almost any dataset
Highly scalable; and relies on Numpy for parallel processing
'''

import numpy as np

# NaiveBayes class: must be intialized with 
# number of data classes, features, smoothing factor
class NaiveBayes:
	
	# initialize
	def __init__(self, classes, features, smoothen):
		self.trained  = False                                # lazy-evaluation: compute terms just before testing begins
		self.smoothen = smoothen                             # smoothing factor for variance
		self.classes  = classes                              # number of classes
		# --- data learned during training ---
		self.freq = np.zeros(classes, np.float)              # class frequency
		self.sum1 = np.zeros((classes, features), np.float)  # sum of feature values per class (for mean)
		self.sum2 = np.zeros((classes, features), np.float)  # sum of square of feature values per class (for variance)
		# --- data needed during testing ---
		self.term = np.empty(classes, np.float)              # constant term per class
		self.mean = np.zeros((classes, features), np.float)  # expected feature values per class
		self.varn = np.ones ((classes, features), np.float)  # augmented variance of feature value per class

	# highly scalable training operation:
	# 1. does not copy the training sample's data -- only adds them to current values
	# 2. does not compute means and variances immediately -- lazy evaluation (see below)
	def train(self, label, values):
		self.trained     = False                             # mark self as "untrained"
		self.freq[label] += 1                                # increment class frequency
		self.sum1[label] += values                           # add feature values to sum1
		self.sum2[label] += np.square(values)                # add squares of value to sum2
	
	# lazy evaluation: compute means and variances just before testing begins.
	# speeds up train() calls by postponing expensive operations like div, log, etc
	def finalize(self):
		# for each class
		for k in range(self.classes):
			# if at least one sample was seen for this class
			if self.freq[k] > 0:
				# mean value of feature, μ = Σx / N
				self.mean[k] = (self.sum1[k] / self.freq[k])
				# variance of feature, σ^2 = Σ(x^2) / N - μ^2
				# augmented variance, z = 2*σ^2 + smoothing_factor
				self.varn[k] = 2*(self.sum2[k] / self.freq[k] - self.mean[k]**2) + self.smoothen
				# class constant  = -log(N) + Σ(z)/2
				self.term[k] = - np.log(self.freq[k]) + np.sum(np.log(self.varn[k]))/2
			else:
				# if no samples were seen, set class_constant to infinity
				self.term[k] = float("inf")
		# mark self as trained
		self.trained = True

	# returns the class label predicted for the given input values
	def predict(self, values):
		# lazy evaluation: train once before testing begins
		if not self.trained: self.finalize()
		# find the probability multiplier for each class (see report)
		# predicted class is the class with the smallest multiplier
		return np.argmin([
			# prob[k] = class_const[k] + Σ (x-μ)^2/z
			self.term[k] + np.sum( ((values - self.mean[k])**2) / self.varn[k]) 
			# for all classes
			for k in range(self.classes)
		])

