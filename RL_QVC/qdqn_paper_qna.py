from msilib import sequence
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq 
import sympy



from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer

import tqdm


def eval_model(env, model, pool):
    print("=======================")
    correctly_answered = 0.0
    print(model)
    for sample, index in tqdm.tqdm(pool, desc="Evaluating"):
        obs = env.reset(sample)
        state = None
        done = False
        sequence = []
        
        while not done:
            action = model.predict(obs)
            sequence.append(action)
            obs, _, done, info = env.step(action)

        if info["selected_choice"] == sample.answer:
            correctly_answered += 1
            
        print(sequence,info["selected_choice"] == sample.answer )
            
    rewards.append(correctly_answered/len(pool))

    print("Correct: ", correctly_answered)
    print("Out of: ", len(pool))
    return correctly_answered/len(pool)



class QDQN(object):
    def __init__(self, action_space, state_space) -> None:
        super().__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.qubits = [cirq.GridQubit(0, i) for i in range(4)]
        self.q_network = self.make_func_approx()
        self.learning_rate = 0.001
        self.opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.buff = 1000
        self.batch = 32
        self.states = np.zeros((self.buff, self.state_space))
        self.actions = np.zeros((self.buff, 1))
        self.rewards = np.zeros((self.buff, 1))
        self.dones = np.zeros((self.buff, 1))
        self.next_states = np.zeros((self.buff, self.state_space))
        # Q Learning
        self.gamma = 0.99 
        self.epsilon = 1.0
        self.epsilon_decay = 0.999 #need to change
        self.epsilon_min = 0.01
        self.counter = 0

    def make_func_approx(self):
        readout_operators = [cirq.Z(self.qubits[i]) for i in range(2,4)]

        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        diff = tfq.differentiators.ParameterShift()
        init = tf.keras.initializers.Zeros
        pqc = tfq.layers.PQC(self.make_circuit(self.qubits), readout_operators, differentiator=diff, initializer=init)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=pqc)
        return model

    def convert_data(self, classical_data, flag=True):
        ops = cirq.Circuit()
        for i, ang in enumerate(classical_data):
            print(ang)
            ang = 0 if ang < 0 else 1
            ops.append(cirq.rx(np.pi * ang).on(self.qubits[i]))
            ops.append(cirq.rz(np.pi * ang).on(self.qubits[i]))
        if flag:
            return tfq.convert_to_tensor([ops])
        else:
            return ops

    def one_qubit_unitary(self, bit, symbols):
        return cirq.Circuit(
            cirq.X(bit)**symbols[0],
            cirq.Y(bit)**symbols[1],
            cirq.Z(bit)**symbols[2])

    def two_qubit_pool(self, source_qubit, sink_qubit, symbols):
        pool_circuit = cirq.Circuit()
        sink_basis_selector = self.one_qubit_unitary(sink_qubit, symbols[0:3])
        source_basis_selector = self.one_qubit_unitary(source_qubit, symbols[3:6])
        pool_circuit.append(sink_basis_selector)
        pool_circuit.append(source_basis_selector)
        pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
        pool_circuit.append(sink_basis_selector**-1)
        return pool_circuit

    def make_circuit(self, qubits):
        m = cirq.Circuit()
        symbols = sympy.symbols('q0:48') # 4 qubits * 3 weights per bit * 3 layers + 2 * 6 pooling = 36 + 12 = 48
        m += self.layer(symbols[:12], qubits)
        m += self.layer(symbols[12:24], qubits)
        m += self.layer(symbols[24:36], qubits)
        #print(m)
        m += self.two_qubit_pool(self.qubits[0], self.qubits[2], symbols[36:42])
        m += self.two_qubit_pool(self.qubits[1], self.qubits[3], symbols[42:])
        print(m)
        return m
    
    def layer(self, weights, qubits):
        l = cirq.Circuit()
        for i in range(len(qubits) - 1):
            l.append(cirq.CNOT(qubits[i], qubits[i+1]))
        l.append([cirq.Moment([cirq.rx(weights[j]).on(qubits[j]) for j in range(4)])])
        l.append([cirq.Moment([cirq.ry(weights[j + 4]).on(qubits[j]) for j in range(4)])])
        l.append([cirq.Moment([cirq.rz(weights[j + 8]).on(qubits[j]) for j in range(4)])])
        return l
    
    def remember(self, state, action, reward, next_state, done):
        i = self.counter % self.buff
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = int(done)
        self.counter += 1

    def get_action(self, obs):
    
        if random.random() < self.epsilon: 
            return np.random.choice(self.action_space)
        else:
            print(self.q_network.predict(self.convert_data(obs), steps=1),np.argmax(self.q_network.predict(self.convert_data(obs), steps=1)))
            return np.argmax(self.q_network.predict(self.convert_data(obs), steps=1))
        
    def predict(self,obs):
        pred = np.argmax(self.q_network.predict(self.convert_data(obs), steps=1))
        return pred


    #@tf.function
    def train(self):
        batch_indices = np.random.choice(min(self.counter, self.buff), self.batch)
        state_batch = tfq.convert_to_tensor([self.convert_data(i, False) for i in self.states[batch_indices]])
        action_batch = tf.convert_to_tensor(self.actions[batch_indices], dtype=tf.int32)
       
        action_batch = [[i, action_batch[i][0]] for i in range(action_batch.shape[0])]
        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)
        dones_batch = tf.convert_to_tensor(self.dones[batch_indices], dtype=tf.float32)
        next_state_batch = tfq.convert_to_tensor([self.convert_data(i, False) for i in self.next_states[batch_indices]])

        with tf.GradientTape() as tape:
            next_q = self.q_network(next_state_batch)
            next_q = tf.expand_dims(tf.reduce_max(next_q, axis=1), -1)
            y = reward_batch + (1 - dones_batch) * self.gamma * next_q
            q_guess = self.q_network(state_batch, training=True)
            pred = tf.gather_nd(q_guess, action_batch)
            pred = tf.reshape(pred, [self.batch, 1])
            msbe = tf.math.reduce_mean(tf.math.square(y - pred))

        grads = tape.gradient(msbe, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

iterations = 10000
rolling_avg = 200
learn_delay = 500

# data pool
data_pool = QASC.prepare(split="train")
val_pool = QASC.prepare(split="val")

# featurizer
featurizer = InformedFeaturizer()
#featurizer = SimpleFeaturizer()


from nlp_gym.envs.common.observation import BaseObservation
from nlp_gym.envs.common.reward import RewardFunction


class MyRewardFunction(RewardFunction):
    def __call__(self, observation: BaseObservation, action: str, target: str) -> float:

        current_time_step = observation.get_current_time_step()
        total_time_steps = observation.get_total_steps()
        selected_choice = observation.get_last_choice()        
        # 1 * (2 ^ index)
        
        correct = 0
        # if current action is ANSWER or ran out of input, then check the current choice and produce terminal reward
        if action == "ANSWER" or current_time_step == total_time_steps - 1:
            reward = (2.0 * (2 ^ current_time_step)) if selected_choice == target else 0.0
            if selected_choice == target:
              correct = 1 
        elif action == "CONTINUE" and selected_choice != target:
            reward = (1.0 * (2 ^ current_time_step)) if selected_choice != target else 0.0
        else:
            reward = 0.0
        return [reward,correct]

# [1,1,0] 
#rewards 
# [1*2^0,1*2^1,2*2^2]



# seq tag env
env = QAEnv(observation_featurizer=featurizer)
for sample, weight in data_pool:
    env.add_sample(sample, weight)


agent = QDQN(env.action_space.n, env.observation_space.shape[0])
rewards = []
avg_reward = deque(maxlen=iterations)
best_avg_reward = avg = -math.inf
rs = deque(maxlen=rolling_avg)
frames = []

from PIL import Image

for i in range(iterations):
    s1 = env.reset()
    total_reward = 0
    done = False
    steps = 0
    sequence = []
    r2 = 0
    rewardsequence= []
    
    while not done:
        #im = Image.fromarray(env.render(mode='rgb_array'))
        #frames.append(im)
        action = agent.get_action(s1)
        steps += 1
        s2, reward, done, _ = env.step(action)
        total_reward += reward
        r2 += reward
        sequence.append(action)
        rewardsequence.append(reward)
        agent.remember(s1, action, reward, s2, done)
        if agent.counter > learn_delay and done:
            agent.train()
        if done:
            rewards.append(total_reward/steps)
            rs.append(total_reward/steps)
        else:
            s1 = s2
    avg = np.mean(rs)
    avg_reward.append(avg)
    if avg > best_avg_reward:
        best_avg_reward = avg
    print("\rEpisode {}/{} || Best average reward {}, Current Avg {}, Current Iteration Reward {}, eps {}, sequence{}".format(i, iterations, best_avg_reward, avg, total_reward, agent.epsilon,sequence), end='\n', flush=True)


plt.ylim(0, 1)
plt.plot(rewards, color='blue', alpha=0.2, label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Iteration')
plt.show()



import pickle
with open('rewards.pkl', 'wb') as f:
    pickle.dump(rewards, f)
        
print("empty")
print(eval_model(env, agent, val_pool))

env.close()

''' frames[1].save('./images/gym_CartPole.gif',
                save_all=True, append_images=frames[2:], optimize=False, duration=40, loop=0) '''


