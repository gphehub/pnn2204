"""
Versions:
main2204_2Para_load_6R.py:
Based on main2204_2Para_load_5R.py,
modified to run network2Para_load_6R.py.
Modifications are indicated by '#mod_LD6R'

main2204_2Para_load_5R.py:
Based on main2204_2Para_load_R.py,
modified to run network2Para_load_5R.py.
Modifications are indicated by '#mod_LD5R'

main2204_2Para_load_R.py:
Based on main2204_2Para_load.py and main2204_2Para4R.py,
modified to run network2Para_load_R.py.
Modifications are indicated by '#mod_LDR'

main2204_2Para_load.py:
Based on main2204_2Para3a.py, modified to run network2Para_load.py.
Modifications are indicated by '#mod_LD'

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

import network2Para_load_6R #mod_LDR #mod_LD5R #mod_LD6R
net_load = network2Para_load_6R.Network() #mod_LDR #mod_LD5R #mod_LD6R
net_load.SGDmod(evaluation_data=validation_data) #mod_LDR

# import joblib #mod_LD
# joblib.dump(net,'network2Para_load.pkl') #mod_2a #mod_2b #mod_3a #mod_LD
