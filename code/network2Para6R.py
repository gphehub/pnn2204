"""
Versions:
network2Para6R.py:
Based on network2Para5R.py and network2Para_load_6R.py.
Added saved_weight_loader.
Modifications are indicated by '#mod_6R'.

network2Para5R.py:
Based on network2Para4R.py and network2Para3a.py.
Changed the way of saving the biases and weights to files.
Modifications are indicated by '#mod_5R'

network2Para4R.py:
Based on network2Para4a.py.
Changed the activation function to ReLU, and replaced "sigmoid" with "activation_fn".
Added commands to save the initial values of biases and weights to file.
Modifications are indicated by '#mod_4R'

network2Para4a.py:
Based on network2Para3a.py.
Added "def identical_weight_initializer(self)".
Deleted "def save(self, filename)" and "def load(filename)".
Modifications are indicated by '#mod_4a'

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

import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * activation_fn_prime(z) #mod_4R


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizesNN1, sizesNN2,
                 costNN1=CrossEntropyCost, costNN2=CrossEntropyCost): #mod_1a
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layersNN1 = len(sizesNN1) #mod_1a
        self.sizesNN1 = sizesNN1 #mod_1a
        self.num_layersNN2 = len(sizesNN2) #mod_1a
        self.sizesNN2 = sizesNN2 #mod_1a

        self.default_weight_initializer()

        self.costNN1=costNN1 #mod_1a
        self.costNN2=costNN2 #mod_1a

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biasesNN1 = [np.random.randn(y, 1) for y in self.sizesNN1[1:]] #mod_1a
        self.weightsNN1 = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizesNN1[:-1], self.sizesNN1[1:])] #mod_1a
        self.biasesNN2 = [np.random.randn(y, 1) for y in self.sizesNN2[1:]] #mod_1a
        self.weightsNN2 = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizesNN2[:-1], self.sizesNN2[1:])] #mod_1a

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biasesNN1 = [np.random.randn(y, 1) for y in self.sizesNN1[1:]] #mod_1a
        self.weightsNN1 = [np.random.randn(y, x)
                        for x, y in zip(self.sizesNN1[:-1], self.sizesNN1[1:])] #mod_1a
        self.biasesNN2 = [np.random.randn(y, 1) for y in self.sizesNN2[1:]] #mod_1a
        self.weightsNN2 = [np.random.randn(y, x)
                        for x, y in zip(self.sizesNN2[:-1], self.sizesNN2[1:])] #mod_1a

    def identical_weight_initializer(self): #mod_4a
        """Only work for the parallel connection of two identical networks.
        """
        self.biasesNN2 = self.biasesNN1
        self.weightsNN2 = self.weightsNN1
        
    def saved_weight_loader(self, filename): #mod_6R
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

    def SGD(self, training_data, epochs, epochs_sep, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False): #mod_1a
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        self.save('biases&weights_initialized.pickle') #mod_5R

        if evaluation_data:
            n_data = len(evaluation_data)
            maxNN1 = 0 #mod_1a
            maxNN2 = 0 #mod_1a
            maxPara = 0 #mod_1a
            EmaxNN1 = 0 #mod_1a
            EmaxNN2 = 0 #mod_1a
            EmaxPara = 0 #mod_1a
            with open('classification_accuracy.csv','w', newline='') as resultsave: #mod_3a
                writer=csv.writer(resultsave) #mod_3a
                writer.writerow(('Epoch','NN1','NN1p','NN2','NN2p','Para')) #mod_3a
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data),
                    epochs, epochs_sep, j) #mod_1a
            """ #mod_1a
            print("Epoch %s training complete: " % j)
            if monitor_training_cost:
                cost = self.total_costNN1(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_costNN1(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            """
            if monitor_evaluation_accuracy:
                """ #mod_1a
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
                """
                rateNN1 = self.accuracyNN1(evaluation_data)/n_data #mod_1a
                rateNN2 = self.accuracyNN2(evaluation_data)/n_data #mod_1a
                rateNN1p = self.accuracyNN1p(evaluation_data)/n_data #mod_2a
                rateNN2p = self.accuracyNN2p(evaluation_data)/n_data #mod_2a
                ratePara = self.accuracyPara(evaluation_data)/n_data #mod_1a
                print("Epoch {0}: NN1={1}, NN1p={2}, NN2={3}, NN2p={4}, Para={5}".format(
                    j, rateNN1, rateNN1p, rateNN2, rateNN2p, ratePara)) #mod_1a #mod_2a

                # Saving the parameters: #mod_1a
                with open('classification_accuracy.csv','a', newline='') as resultsave: #mod_3a
                    writer=csv.writer(resultsave) #mod_3a
                    writer.writerow((j, rateNN1, rateNN1p, rateNN2, rateNN2p, ratePara)) #mod_3a

                if j < epochs_sep:
                    if rateNN1 > maxNN1:
                        maxNN1 = rateNN1
                        EmaxNN1 = j

                        data1 = {"sizes": self.sizesNN1,
                                 "biases": [b.tolist() for b in self.biasesNN1],
                                 "weights": [w.tolist() for w in self.weightsNN1]} #mod_5R
                        f1 = open('biases&weights_network1.pickle', "w") #mod_5R
                        json.dump(data1, f1) #mod_5R
                        f1.close() #mod_5R
                    if rateNN2 > maxNN2:
                        maxNN2 = rateNN2
                        EmaxNN2 = j

                        data2 = {"sizes": self.sizesNN2,
                                 "biases": [b.tolist() for b in self.biasesNN2],
                                 "weights": [w.tolist() for w in self.weightsNN2]} #mod_5R
                        f2 = open('biases&weights_network2.pickle', "w") #mod_5R
                        json.dump(data2, f2) #mod_5R
                        f2.close() #mod_5R
                else:
                    if ratePara > maxPara:
                        maxPara = ratePara
                        EmaxPara = j
                        self.save('biases&weights_optimized.pickle') #mod_5R
                # Saving completed.

            else:
                print("Epoch {0} complete".format(j))
        print('\nmax(NN1)=',maxNN1,'at Epoch',EmaxNN1,
              '\nmax(NN2)=',maxNN2,'at Epoch',EmaxNN2,
              '\nmax(Para)=',maxPara,'at Epoch',EmaxPara) #mod_1a
        with open('classification_accuracy.csv','a', newline='') as resultsave: #mod_3a
            writer=csv.writer(resultsave) #mod_3a
            writer.writerow(('max(NN1)=',maxNN1,'at Epoch',EmaxNN1)) #mod_3a
            writer.writerow(('max(NN2)=',maxNN2,'at Epoch',EmaxNN2)) #mod_3a
            writer.writerow(('max(Para)=',maxPara,'at Epoch',EmaxPara)) #mod_3a

            #print() #mod_1a
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n, epochs, epochs_sep, j): #mod_1a
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_bNN1 = [np.zeros(b.shape) for b in self.biasesNN1] #mod_1a
        nabla_wNN1 = [np.zeros(w.shape) for w in self.weightsNN1] #mod_1a
        nabla_bNN2 = [np.zeros(b.shape) for b in self.biasesNN2] #mod_1a
        nabla_wNN2 = [np.zeros(w.shape) for w in self.weightsNN2] #mod_1a
        for x, y in mini_batch:
            delta_nabla_bNN1, delta_nabla_wNN1, delta_nabla_bNN2, delta_nabla_wNN2 \
                              = self.backprop(x, y, epochs, epochs_sep, j) #mod_1a
            nabla_bNN1 = [nb+dnb for nb, dnb in zip(nabla_bNN1, delta_nabla_bNN1)] #mod_1a #last term of Eq.(94)
            nabla_wNN1 = [nw+dnw for nw, dnw in zip(nabla_wNN1, delta_nabla_wNN1)] #mod_1a #last term of Eq.(93)
            nabla_bNN2 = [nb+dnb for nb, dnb in zip(nabla_bNN2, delta_nabla_bNN2)] #mod_1a #last term of Eq.(94)
            nabla_wNN2 = [nw+dnw for nw, dnw in zip(nabla_wNN2, delta_nabla_wNN2)] #mod_1a #last term of Eq.(93)
        self.weightsNN1 = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weightsNN1, nabla_wNN1)] #mod_1a #Eq.(93)
        self.biasesNN1 = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biasesNN1, nabla_bNN1)] #mod_1a #Eq.(94)
        self.weightsNN2 = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weightsNN2, nabla_wNN2)] #mod_1a #Eq.(93)
        self.biasesNN2 = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biasesNN2, nabla_bNN2)] #mod_1a #Eq.(94)

    def backprop(self, x, y, epochs, epochs_sep, j): #mod_1a
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_bNN1 = [np.zeros(b.shape) for b in self.biasesNN1] #mod_1a
        nabla_wNN1 = [np.zeros(w.shape) for w in self.weightsNN1] #mod_1a
        # feedforward
        activationNN1 = x #mod_1a
        activationsNN1 = [x] # list to store all the activations, layer by layer #mod_1a
        zsNN1 = [] # list to store all the z vectors, layer by layer #mod_1a
        for b, w in zip(self.biasesNN1, self.weightsNN1): #mod_1a
            zNN1 = np.dot(w, activationNN1)+b #mod_1a
            zsNN1.append(zNN1) #mod_1a
            activationNN1 = activation_fn(zNN1) #mod_1a #mod_4R
            activationsNN1.append(activationNN1) #mod_1a

        nabla_bNN2 = [np.zeros(b.shape) for b in self.biasesNN2] #mod_1a
        nabla_wNN2 = [np.zeros(w.shape) for w in self.weightsNN2] #mod_1a
        # feedforward
        activationNN2 = x #mod_1a
        activationsNN2 = [x] # list to store all the activations, layer by layer #mod_1a
        zsNN2 = [] # list to store all the z vectors, layer by layer #mod_1a
        for b, w in zip(self.biasesNN2, self.weightsNN2): #mod_1a
            zNN2 = np.dot(w, activationNN2)+b #mod_1a
            zsNN2.append(zNN2) #mod_1a
            activationNN2 = activation_fn(zNN2) #mod_1a #mod_4R
            activationsNN2.append(activationNN2) #mod_1a

        # backward pass
        if j < epochs_sep: #mod_1a
            connect=0 #mod_1a
        else: #mod_1a
            connect=1 # Connect the 2 networks when j >= epochs_sep #mod_1a

        activationsNN1[-1]=activation_fn(zsNN1[-1]+connect*zsNN2[-1]) #mod_1a #mod_4R
        activationsNN2[-1]=activation_fn(zsNN2[-1]+connect*zsNN1[-1]) #mod_1a #mod_4R


        deltaNN1 = (self.costNN1).delta(zsNN1[-1]+connect*zsNN2[-1],
                                        activationsNN1[-1], y) #mod_1a Beware of the meaning of "@"! #Eq.(BP3)
        deltaNN2 = (self.costNN2).delta(zsNN2[-1]+connect*zsNN1[-1],
                                        activationsNN2[-1], y) #mod_1a Beware of the meaning of "@"! #Eq.(BP3)

        nabla_bNN1[-1] = deltaNN1 #mod_1a
        nabla_wNN1[-1] = np.dot(deltaNN1, activationsNN1[-2].transpose()) #mod_1a #Eq.(BP4)
        nabla_bNN2[-1] = deltaNN2 #mod_1a
        nabla_wNN2[-1] = np.dot(deltaNN2, activationsNN2[-2].transpose()) #mod_1a #Eq.(BP4)

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layersNN1): #mod_1a
            zNN1 = zsNN1[-l] #mod_1a
            spNN1 = activation_fn_prime(zNN1) #mod_1a #mod_4R
            deltaNN1 = np.dot(self.weightsNN1[-l+1].transpose(), deltaNN1) * spNN1 #mod_1a #mod_1a #point 4 of Sec. 2.6
            nabla_bNN1[-l] = deltaNN1 #mod_1a #mod_1a #point 5 of Sec. 2.6
            nabla_wNN1[-l] = np.dot(deltaNN1, activationsNN1[-l-1].transpose()) #mod_1a #mod_1a #point 5 of Sec. 2.6
        for l in range(2, self.num_layersNN2): #mod_1a
            zNN2 = zsNN2[-l] #mod_1a
            spNN2 = activation_fn_prime(zNN2) #mod_1a #mod_4R
            deltaNN2 = np.dot(self.weightsNN2[-l+1].transpose(), deltaNN2) * spNN2 #mod_1a #mod_1a #point 4 of Sec. 2.6
            nabla_bNN2[-l] = deltaNN2 #mod_1a #mod_1a #point 5 of Sec. 2.6
            nabla_wNN2[-l] = np.dot(deltaNN2, activationsNN2[-l-1].transpose()) #mod_1a #mod_1a #point 5 of Sec. 2.6
        return (nabla_bNN1, nabla_wNN1, nabla_bNN2, nabla_wNN2) #mod_1a

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
            results = [(np.argmax(self.feedforwardPara(x)), np.argmax(y))
                       for (x, y) in data] #mod_1a
        else:
            results = [(np.argmax(self.feedforwardPara(x)), y)
                        for (x, y) in data] #mod_1a
        return sum(int(x == y) for (x, y) in results)



    def total_costNN1(self, data, lmbda, convert=False): #mod_1a
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforwardNN1(x) #mod_1a
            if convert: y = vectorized_result(y)
            cost += self.costNN1.fn(a, y)/len(data) #mod_1a
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weightsNN1) #mod_1a
        return cost

    def save(self, filename): #mod_5R
        """Save the neural network to the file ``filename``."""
        data = {"sizes1": self.sizesNN1,
                "biases1": [b.tolist() for b in self.biasesNN1],
                "weights1": [w.tolist() for w in self.weightsNN1],
                "sizes2": self.sizesNN2,
                "biases2": [b.tolist() for b in self.biasesNN2],
                "weights2": [w.tolist() for w in self.weightsNN2]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
# def load(filename): #mod_4a

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

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
