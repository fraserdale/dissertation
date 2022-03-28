
#import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf


tf.get_logger().setLevel('ERROR')
tf.enable_eager_execution()

def eval_model(env,  pool):
    correctly_answered = 0.0
    choices = {
        "A":0,
        "B":0,
        "C":0,
        "D":0,
        "E":0,
        "F":0,
        "G":0,
        "H":0
    }
    for sample, index in tqdm.tqdm(pool, desc="Evaluating"):
        ''' print(sample)
        exit()
        obs = env.reset(sample)
        state = None
        done = False
        sequence = []
        
        while not done:
            action = np.random.choice(env.action_space.n)
            sequence.append(action)
            obs, _, done, info = env.step(action)

        if info["selected_choice"] == sample.answer:
            correctly_answered += 1 '''
        choices[sample.answer] += 1
    print(choices)
    return [correctly_answered,len(pool)]



from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer

import tqdm
# data pool
data_pool = QASC.prepare(split="train")
val_pool = QASC.prepare(split="val")

# featurizer
featurizer = InformedFeaturizer()
# seq tag env
env = QAEnv(observation_featurizer=featurizer)
for sample, weight in data_pool:
    env.add_sample(sample, weight) 

if __name__ == "__main__":
    total_correct = 0
    total_answered = 0  
    for n in range (100):
        env.reset()
        ans = eval_model(env,val_pool)
        total_correct += ans[0]
        total_answered += ans[1]

    print("All")
    print(total_correct/total_answered)