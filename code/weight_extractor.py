"""
Versions:
Based on network2Para_load_6S.py and networkPara4a.py.
For extracting the weights at the output layer.
"""

#### Libraries
# Standard library
import csv #mod_1a
import pickle #mod_1a

import json #mod_LD5R
import numpy as np

f = open('biases&weights_optimized.pickle', "r")
data = json.load(f)
f.close()
# net = Network(data["sizes1"], data["sizes2"], costNN1=costNN1, costNN2=costNN2) #mod_1a #mod_LD5R
sizesNN1 = data["sizes1"] #mod_LD5R
biasesNN1 = [np.array(b) for b in data["biases1"]] #mod_1a #mod_LD5R
weightsNN1 = [np.array(w) for w in data["weights1"]] #mod_1a #mod_LD5R
sizesNN2 = data["sizes2"] #mod_LD5R
biasesNN2 = [np.array(b) for b in data["biases2"]] #mod_1a #mod_LD5R
weightsNN2 = [np.array(w) for w in data["weights2"]] #mod_1a #mod_LD5R

with open('weights_at_the_output_layer.csv','w', newline='') as weights_save:
    writer=csv.writer(weights_save)
    writer.writerow(["Weights of Network 1:"])
    writer.writerows(weightsNN1[-1])
    writer.writerow(["Weights of Network 2:"])
    writer.writerows(weightsNN2[-1])

print("Weights exported to file weights_at_the_output_layer.csv.")
