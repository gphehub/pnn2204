"""
Versions:
main2204_2Para6T.py:
Based on main2204_2Para6S.py, modified to run network2Para6T.py.
Included main2204_2Para_load_6T.py.
Modifications are indicated by '#mod_6T'.

main2204_2Para6S.py:
Based on main2204_2Para5S.py, modified to run network2Para6S.py.
Included main2204_2Para_load_6S.py.
Modifications are indicated by '#mod_6S'

main2204_2Para5S.py:
Based on main2204_2Para4S.py, modified to run network2Para5S.py.
Modifications are indicated by '#mod_5S'

main2204_2Para4S.py:
Based on main2204_2Para4R.py, modified to run network2Para4S.py.
Modifications are indicated by '#mod_4S'

main2204_2Para4R.py:
Based on main2204_2Para4a.py, modified to run network2Para4R.py.
Modifications are indicated by '#mod_4R'

main2204_2Para4a.py:
Based on main2204_2Para3a2.py.
Modified to run network2Para4a.py.
Added "net.identical_weight_initializer()".
Included main2204_2Para_load.py.
Modifications are indicated by '#mod_4a'

main2204_2Para3a2.py:
Based on main2204_2Para3a.py, disabled "net.large_weight_initializer()".
Modifications are indicated by '#mod_3a2'

main2204_2Para3a.py:
Based on main2204_2Para2b.py, modified to run network2Para3a.py.
Modifications are indicated by '#mod_3a'

main2204_2Para2b.py:
Based on main2204_2Para2a.py, modified to run network2Para2b.py.
Modifications are indicated by '#mod_2b'

main2204_2Para2a.py:
Based on main2204_2Para1a.py, modified to run network2Para2a.py.
Modifications are indicated by '#mod_2a'

main2204_2Para1a.py:
Based on main_network2.4.py, modified to run network2Para1a.py.
Modifications are indicated by '#mod_1a'
"""

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2Para6T #mod_1a #mod_2a #mod_2b #mod_3a #mod_4a #mod_4R #mod_4S #mod_5S #mod_6S #mod_6T

net = network2Para6T.Network([784,48,35,10],[784,48,35,10],
                             costNN1=network2Para6T.CrossEntropyCost,
                             costNN2=network2Para6T.CrossEntropyCost) #mod_1a #mod_2a #mod_2b #mod_3a #mod_4a #mod_4R #mod_4S #mod_5S #mod_6S #mod_6T

# net.large_weight_initializer() #mod_3a2

# net.identical_weight_initializer() #mod_4a
"""Only work for the parallel connection of two identical networks.
"""

# net.saved_weight_loader('biases&weights_initialized.pickle') #mod_6S

net.SGD(training_data, 100, 0,
        10, 0.1,
        lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True) #mod_1a
"""
Format: SGD(self, training_data, epochs, epochs_sep,
            mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False)
"""

import joblib
joblib.dump(net,'network2Para6T.pkl') #mod_2a #mod_2b #mod_3a #mod_4a #mod_4R #mod_4S #mod_5S #mod_6S #mod_6T



print("") #mod_4R

# Included main2204_2Para_load_6T.py:
import network2Para_load_6T #mod_4a #mod_4R #mod_4S #mod_5S #mod_6S #mod_6T
net_load = network2Para_load_6T.Network() #mod_4a #mod_4R #mod_4S #mod_5S #mod_6S #mod_6T
net_load.SGDmod(evaluation_data=validation_data) #mod_4a #mod_4S #mod_4S
