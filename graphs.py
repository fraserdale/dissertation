
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
for x in range(len(objectsC[0])):
    if objectsC[0][x]['answer'] == objectsQ[0][x]['answer'] and objectsC[0][x]['answer'] == "H":
        same += 1
        if  objectsC[0][x]['correct']:
            print("correct: ",str(x))
            sameCorrect[objectsC[0][x]['answer']] += 1
        else:
            print("incorrect: ",str(x))
            sameIncorrect[objectsC[0][x]['answer']] += 1
   
print(same)
print(sameCorrect)
#
x_pos = [i for i, _ in enumerate(choices.keys())]


x = np.arange(len(choices.keys()))  # the label locations
width = 0.2

sameIncorrect = {'A': 55, 'B': 42, 'C': 24, 'D': 15, 'E': 11, 'F': 9, 'G': 9, 'H': 321}
sameCorrect = {'A': 67, 'B': 43, 'C': 58, 'D': 50, 'E': 48, 'F': 41, 'G': 56, 'H': 77}

fig, ax = plt.subplots()
''' bars1 = ax.bar(x - width, sameIncorrect.values(), width, label='Incorrect Prediction',color='red')
bars2 = ax.bar(x , sameCorrect.values(), width, label='Correct Prediction', color='green') '''
bars3 = ax.bar(x, baselines.values(), 0.8, label='Actual Answer', color='blue')

''' ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3) '''
ax.bar_label(bars3, padding=3)

#plt.bar(x_pos, correct.values(), color='blue', alpha=0.2, label='Occurances')
plt.legend()
plt.ylabel('Occurrences')
plt.xlabel('Answer')
plt.xticks(x_pos, correct.keys())
plt.show()
plt.savefig("currentGraph")