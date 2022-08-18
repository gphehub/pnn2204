"""
Versions:
network2Para_load_6R.py:
Based on network2Para_load_5R.py and network2Para_load_6S.py.
Changed "feedforwardNN1" and "feedforwardNN2" in "all_results" into "feedforwardNN1p"
and "feedforwardNN2p".
Write accuracies to the CSV file.
Modifications are indicated by '#mod_LD6R'.

network2Para_load_5R.py:
Based on network2Para_load_R.py and network2Para3a.py.
Changed the way of loading the the saved biases and weights,
to accompany the files saved by network2Para5R.py.
Modifications are indicated by '#mod_LD5R'

network2Para_load_R.py:
Based on network2Para_load.py.
Changed the activation function to ReLU, and replaced "sigmoid" with "activation_fn".
Replace "default_weight_initializer" with "saved_weight_loader".
Modifications are indicated by '#mod_LDR'

network2Para_load.py
Based on network2Para3a.py.
Load previously saved biases and weights of the trained network for analysis.
Deleted unused functions and parameters.
Modifications are indicated by '#mod_LD'

network2Para3a.py:
Based on network2Para2a.py.
Automatically save the classification accuracy result to CSV file.
Modifications are indicated by '#mod_3a'

network2Para2a.py:
Based on network2Para1a.py.
Added feedforwardNN1p, feedforwardNN2p, accuracyNN1p, accuracyNN2p.
Modifications are indicated by '#mod_2a'

network2Para1a.py:
Based on network2.py, modified for parallel connected networks
following networkPara5a.py.
Modifications are indicated by '#mod_1a'



network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import csv #mod_1a
import pickle #mod_1a

import json #mod_LD5R
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

# class QuadraticCost(object): #mod_LD

# class CrossEntropyCost(object): #mod_LD



#### Main Network class
class Network(object):

    def __init__(self): #mod_1a #mod_LD
        self.saved_weight_loader('biases&weights_optimized.pickle') #mod_LDR #mod_LD5R

    def saved_weight_loader(self, filename): #mod_LD5R
        """Load a neural network from the file ``filename``.  Returns an
        instance of Network.
        """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        # net = Network(data["sizes1"], data["sizes2"], costNN1=costNN1, costNN2=costNN2) #mod_1a #mod_LD5R
        self.sizesNN1 = data["sizes1"] #mod_LD5R
        self.biasesNN1 = [np.array(b) for b in data["biases1"]] #mod_1a #mod_LD5R
        self.weightsNN1 = [np.array(w) for w in data["weights1"]] #mod_1a #mod_LD5R
        self.sizesNN2 = data["sizes2"] #mod_LD5R
        self.biasesNN2 = [np.array(b) for b in data["biases2"]] #mod_1a #mod_LD5R
        self.weightsNN2 = [np.array(w) for w in data["weights2"]] #mod_1a #mod_LD5R
    # return net #mod_LD5R

    def separated_weight_loader(self): #mod_LD5R
        f = open('biases&weights_network1.pickle', "r")
        data = json.load(f)
        f.close()
        self.sizesNN1 = data["sizes"] #mod_LD5R
        self.biasesNN1 = [np.array(b) for b in data["biases"]] #mod_1a #mod_LD5R
        self.weightsNN1 = [np.array(w) for w in data["weights"]] #mod_1a #mod_LD5R
        f = open('biases&weights_network2.pickle', "r")
        data = json.load(f)
        f.close()
        self.sizesNN2 = data["sizes"] #mod_LD5R
        self.biasesNN2 = [np.array(b) for b in data["biases"]] #mod_1a #mod_LD5R
        self.weightsNN2 = [np.array(w) for w in data["weights"]] #mod_1a #mod_LD5R

    # def large_weight_initializer(self): #mod_LD

    def feedforwardNN1(self, a): #mod_1a
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biasesNN1, self.weightsNN1): #mod_1a
            a = activation_fn(np.dot(w, a)+b) #mod_4R
        return a

    def feedforwardNN2(self, a): #mod_1a
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biasesNN2, self.weightsNN2): #mod_1a
            a = activation_fn(np.dot(w, a)+b) #mod_4R
        return a

    def feedforwardNN1p(self, a): #mod_2a
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biasesNN1, self.weightsNN1): #mod_1a
            z=np.dot(w, a)+b #mod_2a
            a = activation_fn(z) #mod_2a #mod_4R
        a = activation_fn(z+self.biasesNN2[-1]) #mod_2a #mod_4R
        return a

    def feedforwardNN2p(self, a): #mod_2a
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biasesNN2, self.weightsNN2): #mod_1a
            z=np.dot(w, a)+b #mod_2a
            a = activation_fn(z) #mod_2a #mod_4R
        a = activation_fn(z+self.biasesNN1[-1]) #mod_2a #mod_4R
        return a

    def feedforwardPara(self, a): #mod_1a
        """Return the output of the network if ``a`` is input."""
        a1=a #mod_1a
        a2=a #mod_1a
        for b, w in zip(self.biasesNN1, self.weightsNN1): #mod_1a
            z1=np.dot(w, a1)+b #mod_1a
            a1 = activation_fn(z1) #mod_1a #mod_4R
        for b, w in zip(self.biasesNN2, self.weightsNN2): #mod_1a
            z2=np.dot(w, a2)+b #mod_1a
            a2 = activation_fn(z2) #mod_1a #mod_4R
        a=activation_fn(z1+z2) #mod_1a #mod_4R
        return a

    def SGDmod(self, evaluation_data=None): #mod_LD #mod_LDR

        n_data = len(evaluation_data)
        """ #mod_LD
        with open('classification_accuracy.csv','w', newline='') as resultsave: #mod_3a
            writer=csv.writer(resultsave) #mod_3a
            writer.writerow(('Epoch','NN1','NN1p','NN2','NN2p','Para')) #mod_3a
        """ #mod_LD

        # for j in range(epochs): #mod_LD
        rateNN1 = self.accuracyNN1(evaluation_data)/n_data #mod_1a
        rateNN2 = self.accuracyNN2(evaluation_data)/n_data #mod_1a
        rateNN1p = self.accuracyNN1p(evaluation_data)/n_data #mod_2a
        rateNN2p = self.accuracyNN2p(evaluation_data)/n_data #mod_2a
        ratePara = self.accuracyPara(evaluation_data)/n_data #mod_1a
        print("Accuracy: NN1={0}, NN1p={1}, NN2={2}, NN2p={3}, Para={4}".format(
            rateNN1, rateNN1p, rateNN2, rateNN2p, ratePara)) #mod_1a #mod_2a #mod_LD #mod_LDR

        print("Number of evaluation data: ", n_data) #mod_LD

        # Saving the parameters: #mod_1a
        """ #mod_LD
        with open('classification_accuracy.csv','a', newline='') as resultsave: #mod_3a
            writer=csv.writer(resultsave) #mod_3a
            writer.writerow((j, rateNN1, rateNN1p, rateNN2, rateNN2p, ratePara)) #mod_3a
        """ #mod_LD
        with open('disagreed_results.csv','w', newline='') as resultsave: #mod_LD
            writer=csv.writer(resultsave) #mod_LD
            writer.writerow(("NN1:",rateNN1)) #mod_LD6R
            writer.writerow(("NN1p:",rateNN1p)) #mod_LD6R
            writer.writerow(("NN2:",rateNN2)) #mod_LD6R
            writer.writerow(("NN2p:",rateNN2p)) #mod_LD6R
            writer.writerow(("Para:",ratePara)) #mod_LD6R
            writer.writerow(("")) #mod_LD6S
            writer.writerow(("Number of evaluation data: ", n_data)) #mod_LD
        # Saving completed.

        w1, w2, wb, rb = self.disagreed_results(evaluation_data) #mod_LD

        return wb #mod_LD

    # def update_mini_batch(self, mini_batch, eta, lmbda, n, epochs, epochs_sep, j): #mod_1a #mod_LD
    # def backprop(self, x, y, epochs, epochs_sep, j): #mod_1a #mod_LD

    def disagreed_results(self, data, convert=False): #mod_LD
        if convert:
            all_results = [(x, np.argmax(y), np.argmax(self.feedforwardPara(x)),
                          np.argmax(self.feedforwardNN1p(x)),
                          np.argmax(self.feedforwardNN2p(x)))
                       for (x, y) in data] #mod_LD6R
        else:
            all_results = [(x, y, np.argmax(self.feedforwardPara(x)),
                          np.argmax(self.feedforwardNN1p(x)),
                          np.argmax(self.feedforwardNN2p(x)))
                       for (x, y) in data] #mod_LD6R
        wrong_1 = sum(int(p == y and a != y and b == y) for (x, y, p, a, b) in all_results)
        wrong_2 = sum(int(p == y and a == y and b != y) for (x, y, p, a, b) in all_results)
        wrong_both = sum(int(p == y and a != y and b != y) for (x, y, p, a, b) in all_results)
        right_both = sum(int(p == y and a == y and b == y) for (x, y, p, a, b) in all_results)

        print("Total number of correct results:",
              wrong_1+wrong_2+wrong_both+right_both) #mod_LD
        print("Number of results when both subnetworks are right:",
              right_both) #mod_LD
        print("Number of results when only network1 is wrong:", wrong_1) #mod_LD
        print("Number of results when only network2 is wrong:", wrong_2) #mod_LD
        print("Number of results when both subnetworks are wrong but the combined network is right:",
              wrong_both) #mod_LD

        d_resultsave = open('disagreed_results.csv','a', newline='')
        writer=csv.writer(d_resultsave)
        writer.writerow(("Total number of correct results:",
              wrong_1+wrong_2+wrong_both+right_both))
        writer.writerow(("Number of results when both subnetworks are right:",
              right_both))
        writer.writerow(("Number of results when only network1 is wrong:", wrong_1))
        writer.writerow(("Number of results when only network2 is wrong:", wrong_2))
        writer.writerow(("Number of results when both subnetworks are wrong but the combined network is right:",
              wrong_both))
        writer.writerow((""))
        writer.writerow(("Detailed results when both subnetworks are wrong:",""))
        writer.writerow(("Parallel network", "Network 1", "Network 2"))
        # print("Results: y VS network_para VS network1 VS network2")
        for (x, y, p, a, b) in all_results:
            if p == y and a != y and b != y:
                writer.writerow((p, a, b))
                # print(y, p, a, b)
        d_resultsave.close()
                
        return wrong_1, wrong_2, wrong_both, right_both



    def accuracyNN1(self, data, convert=False): #mod_1a
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforwardNN1(x)), np.argmax(y))
                       for (x, y) in data] #mod_1a
        else:
            results = [(np.argmax(self.feedforwardNN1(x)), y)
                        for (x, y) in data] #mod_1a
        return sum(int(x == y) for (x, y) in results)

    def accuracyNN2(self, data, convert=False): #mod_1a
        if convert:
            results = [(np.argmax(self.feedforwardNN2(x)), np.argmax(y))
                       for (x, y) in data] #mod_1a
        else:
            results = [(np.argmax(self.feedforwardNN2(x)), y)
                        for (x, y) in data] #mod_1a
        return sum(int(x == y) for (x, y) in results)

    def accuracyNN1p(self, data, convert=False): #mod_2a
        if convert:
            results = [(np.argmax(self.feedforwardNN1p(x)), np.argmax(y))
                       for (x, y) in data] #mod_2a
        else:
            results = [(np.argmax(self.feedforwardNN1p(x)), y)
                        for (x, y) in data] #mod_2a
        return sum(int(x == y) for (x, y) in results)

    def accuracyNN2p(self, data, convert=False): #mod_2a
        if convert:
            results = [(np.argmax(self.feedforwardNN2p(x)), np.argmax(y))
                       for (x, y) in data] #mod_2a
        else:
            results = [(np.argmax(self.feedforwardNN2p(x)), y)
                        for (x, y) in data] #mod_2a
        return sum(int(x == y) for (x, y) in results)

    def accuracyPara(self, data, convert=False): #mod_1a
        if convert:
            results = [(np.argmax(self.feedforwardPara(x)), np.argmax(y))
                       for (x, y) in data] #mod_1a
        else:
            results = [(np.argmax(self.feedforwardPara(x)), y)
                        for (x, y) in data] #mod_1a
        return sum(int(x == y) for (x, y) in results)



    # def total_costNN1(self, data, lmbda, convert=False): #mod_1a #mod_LD

    # def save(self, filename): #mod_LD

#### Loading a Network
# def load(filename): #mod_LD

#### Miscellaneous functions
# def vectorized_result(j): #mod_LD

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def activation_fn(z): #mod_4R
    """The sigmoid function."""
    # return 1.0/(1.0+np.exp(-z)) #mod_4R
    """The ReLU function.""" #mod_4R
    return np.where(z<0,0,z) #mod_4R

def activation_fn_prime(z): #mod_4R
    """Derivative of the sigmoid function."""
    # return sigmoid(z)*(1-sigmoid(z)) #mod_4R
    """Derivative of the ReLU function.""" #mod_4R
    return np.where(z<0,0,1) #mod_4R
