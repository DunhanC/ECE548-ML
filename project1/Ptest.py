import math
import operator
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.colors import ListedColormap
from random import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from perceptron import PerceptronClassifier
from sklearn.linear_model import perceptron
from sklearn.ensemble import AdaBoostClassifier

FILENAME = "dataset/num1.dat"

# probability of an example being in the training set
PROBABILITY_TRAINING_SET = 0.5

# learning rate for perceptron
ETA = 0.2

# desired threshold for error rate; 0.2 --> 20%
THRESHOLD = 0.00
# maximum number of epochs for training
UPPER_BOUND = 90
# verbose flag
IS_VERBOSE = True
Run = 10


def split_dataset(examples, prob_training):
	"""
	receives list of examples
	returns a tuple consisting of a training set and testing set
	"""
	training_set = []
	testing_set = []

	# generate a random number [1,100]. if it is greater than
	# prob_training then add the example to the
	# testing set; otherwise add it to the training set
	percent_training = prob_training * 100;
	for example in examples:
		result = randint(1, 100)
		# if the result is a number less than percent_training,
		# add to training set; else add it to the testing set
		if (result < percent_training):
			training_set.append(example)
		else:
			testing_set.append(example)

	return (training_set, testing_set)

def load_dataset(filename):
	"""
	given a filename that points to a file containing the data-set,
	load it into memory and return an array containing this data-set
	"""
	dataset = []
	# open the data-set file
	file = open(filename, "r")
	# we want to load this data-set into a 2D array
	# where each row is an example and each column is
	# an attribute.
	for line in file:
		example = line.strip().split(" ") # a row in the data-set
		dataset.append(list(map(float, example[:]))) # append it to the 2D array

	return dataset

def split_attribute_and_label(dataset):
	"""
	split attribute vectors from their class-labels
	"""

	# add 0.1 because values are processed as floats and we may have 0.999...
	class_labels = [round(row[-1]) for row in dataset]
	attributes = [row[:-1] for row in dataset]
	return (attributes, class_labels)

def calculate_error(class_labels, hypothesis_list):
	"""
	calculates simple error rate on a dataset
	:param class_labels: list of given class-labels
	:param hypothesis_list: list of classifier predictions for examples
	"""
	num_errors = 0
	for i in range(len(class_labels)):
		if class_labels[i] != hypothesis_list[i]:
			num_errors += 1

	return (num_errors / len(class_labels))

class ErrorWrapper:
	def __init__(self, num_classifiers, train_error, test_error, scikit_error):
		self.num_classifiers = num_classifiers
		self.train_error = train_error
		self.test_error = test_error
		self.scikit_error = scikit_error

	def __str__(self):
		return "# of Classifiers {0}, Train Error: {1}, Test Error: {2}, Scikit Error: {3}".format(
			self.num_classifiers, self.train_error, self.test_error, self.scikit_error)


def perceptron_avg_run(avg_num_of_run, training_set, testing_set):
    (train_x, train_y) = split_attribute_and_label(training_set)
    (test_x, test_y) = split_attribute_and_label(testing_set)
    ptraining_error  = []
    perceptron_error = []

    for i in range(avg_num_of_run):
        p = perceptron.Perceptron(max_iter=UPPER_BOUND, verbose=0, random_state=None,
                                  fit_intercept=True, eta0=ETA)
        p.fit(train_x, train_y)
        t_result_list = p.predict(train_x)
        ptraining_error.append(calculate_error(train_y, t_result_list))
        result_list = p.predict(test_x)
        perceptron_error.append(calculate_error(test_y, result_list))

    return sum(perceptron_error) / len(perceptron_error) , sum(ptraining_error) / len(ptraining_error)

def our_avg_run(avg_num_of_run,filename):
    dataset = load_dataset(filename)



    ptraining_error  = []
    perceptron_error = []

    for i in range(avg_num_of_run):
        (training_set, testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)
        testing_set = dataset
        if IS_VERBOSE:
        	print("training set size: %s testing set size: %s num instances: %s" %
                  (len(training_set), len(testing_set), len(dataset)))

        (train_x, train_y) = split_attribute_and_label(training_set)
        (test_x, test_y) = split_attribute_and_label(testing_set)
        p = PerceptronClassifier(ETA, THRESHOLD, UPPER_BOUND, False)
        p.fit(train_x, train_y)
        t_result_list = p.predict(train_x)
        ptraining_error.append(calculate_error(train_y, t_result_list))
        result_list = p.predict(test_x)
        perceptron_error.append(calculate_error(test_y, result_list))
        print(p.weights)

    return sum(perceptron_error) / len(perceptron_error) , sum(ptraining_error) / len(ptraining_error)




#dataset = load_dataset(FILENAME)

#(training_set,testing_set) = split_dataset(dataset, PROBABILITY_TRAINING_SET)
#testing_set = dataset

#if IS_VERBOSE:
# 	print("training set size: %s testing set size: %s num instances: %s" %
#          (len(training_set), len(testing_set), len(dataset)))





our_avg_run = our_avg_run(Run, FILENAME)
print("Training error rate %s" % our_avg_run[1])


print("=========")
print("our perceptron error rate on test: %s" % our_avg_run[0])
print("=========")

#perceptron_avg_error  = perceptron_avg_run(Run, training_set, testing_set)

#print("scikit-learn perceptron training error rate %s" % perceptron_avg_error[0])

#print("scikit-learn perceptron testing error rate %s" % perceptron_avg_error[1])