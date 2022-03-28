
from cProfile import label
from secrets import choice
import numpy as np
import matplotlib.pyplot as plt
baselines = {'A': 121, 'B': 103, 'C': 122, 'D': 111, 'E': 120, 'F': 104, 'G': 135, 'H': 110}
choices = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0}
correct = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0}
correctClassical = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0}
correctQuantum = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0}
incorrect =  {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0}

sameCorrect = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0}
sameIncorrect = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0}

import pickle

objectsC = []
objectsQ = []
with (open("./RL_QVC/reinforcement_eval_15pc_139correct_045_50kitter.pkl", "rb")) as openfile:
    while True:
        try:
            objectsQ.append(pickle.load(openfile))
        except EOFError:
            break
with (open("./nlp/classical_eval.pkl", "rb")) as openfile:
    while True:
        try:
            objectsC.append(pickle.load(openfile))
        except EOFError:
            break
        
same = 0
for x in range(len(objectsQ[0])):
    if objectsQ[0][x]['answer'] == objectsC[0][x]['answer']:
        same += 1
        if  objectsQ[0][x]['correct']:
            sameCorrect[objectsC[0][x]['answer']] += 1
        else:
            sameIncorrect[objectsC[0][x]['answer']] += 1
print(same)
print(sameCorrect)
#
x_pos = [i for i, _ in enumerate(choices.keys())]


x = np.arange(len(choices.keys()))  # the label locations
width = 0.2

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, sameCorrect.values(), width, label='Same Correct Prediction',color='green')
bars2 = ax.bar(x + width/2 , sameIncorrect.values(), width, label='Same Incorrect Prediction', color='red')
#bars3 = ax.bar(x + width, baselines.values(), width, label='Actual Answer', color='blue')

ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3)
#ax.bar_label(bars3, padding=3)

#plt.bar(x_pos, correct.values(), color='blue', alpha=0.2, label='Occurances')
plt.legend()
plt.ylabel('Occurances')
plt.xlabel('Answer')
plt.xticks(x_pos, correct.keys())
plt.show()
plt.savefig("currentGraph")