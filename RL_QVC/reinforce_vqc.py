
#import gym
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf
from policies import * 

tf.get_logger().setLevel('ERROR')
tf.enable_eager_execution()

import pickle

def eval_model(env, model, pool):
    print("=======================")
    correctly_answered = 0.0
    print(model)
    outputidk = []
    c = 0
    for sample, index in tqdm.tqdm(pool, desc="Evaluating"):
        obs = env.reset(sample)
        state = None
        done = False
        sequence = []
        
        while not done:
            action = model.get_action(obs)
            sequence.append(action)
            obs, _, done, info = env.step(action)

        if info["selected_choice"] == sample.answer:
            correctly_answered += 1
            
        outputidk.append({"question":c,"answer":info["selected_choice"], "correct":info["selected_choice"] == sample.answer})
            
        print(sequence,info["selected_choice"] == sample.answer )
        c=c+1      
    with open('reinforcement_eval.pkl', 'wb') as f:
        pickle.dump(outputidk, f)
                  
    print("Correct: ", correctly_answered)
    print("Out of: ", len(pool))
    return correctly_answered/len(pool)

class REINFORCE(object):
    def __init__(self, a_space, o_space):
        self.action_space = a_space
        self.state_space = o_space
        self.gamma = 0.99
        self.states, self.actions, self.rewards = [], [], []
        self.policy = ReUpPolicy(self.state_space, 15, self.action_space)
        #self.policy = NoReUpPolicy(self.state_space, 5, self.action_space)
        self.opt = tf.keras.optimizers.Adam(lr=0.045)

    def remember(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def get_action(self, obs):
        probs = self.policy(np.array([obs])).numpy()[0]
        return np.random.choice(self.action_space, p=probs)

    def discount_rewards(self, rewards):
        d_rewards = np.zeros_like(rewards)
        Gt = 0
        # Discount rewards
        for i in reversed(range(len(rewards))):
            Gt = Gt * self.gamma + rewards[i]
            d_rewards[i] = Gt
        # Normalize
        d_rewards = (d_rewards - np.mean(d_rewards)) / (np.std(d_rewards) + 1e-11)
        return d_rewards

    def update(self):
        state_batch = tf.convert_to_tensor(self.states, dtype=tf.float32)
        action_batch = tf.convert_to_tensor([[i, self.actions[i]] for i in range(len(self.actions))], dtype=tf.int32)
        reward_batch = tf.convert_to_tensor(self.discount_rewards(self.rewards), dtype=tf.float32)

        with tf.GradientTape() as tape:
            model_dist = self.policy(state_batch)
            action_probs = tf.gather_nd(model_dist, action_batch)
            log_probs = tf.math.log(action_probs)
            error = tf.math.reduce_mean(-log_probs * reward_batch)

        grads = tape.gradient(error, self.policy.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()


from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer

import tqdm
# data pool
data_pool = QASC.prepare(split="train")
val_pool = QASC.prepare(split="val")

# featurizer
featurizer = InformedFeaturizer()


''' from nlp_gym.envs.common.observation import BaseObservation
from nlp_gym.envs.common.reward import RewardFunction
class MyRewardFunction(RewardFunction):
    def __call__(self, observation: BaseObservation, action: str, target: str) -> float:

        current_time_step = observation.get_current_time_step()
        total_time_steps = observation.get_total_steps()
        selected_choice = observation.get_last_choice()        
        # 1 * (2 ^ index)
        #print("timestep: ",current_time_step)
        if selected_choice == target:
            max_reward =  (2.0 * (2 ^ current_time_step))
            #print("c1 ",max_reward)
        else: 
            max_reward = (1.0 * (2 ^ current_time_step))
            #print("c2 ",max_reward)

        #print("max out",max_reward) 
        
        correct = 0
        # if current action is ANSWER or ran out of input, then check the current choice and produce terminal reward
        if action == "ANSWER" or current_time_step == total_time_steps - 1:
            reward = 1 if selected_choice == target else 0.0
            if selected_choice == target:
              correct = 1 
        elif action == "CONTINUE" and selected_choice != target:
            reward = 0.1 if selected_choice != target else 0.0
        else:
            reward = 0.0
       
        if reward != 0 and max_reward == 0:
            print("reward/max",reward,max_reward)
        if reward > 0:
            return reward/max_reward
        else:            
            return 0 '''

# seq tag env
env = QAEnv(observation_featurizer=featurizer)
for sample, weight in data_pool:
    env.add_sample(sample, weight) 

if __name__ == "__main__":
    iterations = 20000
    rolling_avg = 2500

    
    #env = gym.make("CartPole-v1")
    agent = REINFORCE(env.action_space.n, env.observation_space.shape[0])
    rewards = []
    avg_reward = deque(maxlen=iterations)
    best_avg_reward = avg = -math.inf
    rs = deque(maxlen=rolling_avg)
        
    
    for i in range(iterations):
        s1 = env.reset()
        total_reward = 0
        done = False
        
        actions = []
        steps = 0
        
        while not done:
            action = agent.get_action(s1)
            actions.append(action)
            s2, reward, done, _ = env.step(action)
            total_reward += reward
            agent.remember(s1, action, reward)
            s1 = s2
            steps +=1 
        agent.update()
        rewards.append(total_reward)
        rs.append(total_reward)
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
        print("\rEpisode {}/{} || Best average reward {}, Current Avg {}, Current Iteration Reward {}, actions {}".format(i, iterations, best_avg_reward, avg, total_reward, actions), flush=True)
    
    plt.plot(rewards, color='blue', alpha=0.2, label='Reward')
    plt.plot(avg_reward, color='red', label='Average')
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.show()
    plt.savefig("reinforce2")
    print(eval_model(env,agent,val_pool))


''' 
layers = 12
lr= 0.075
Correct:  119.0
Out of:  926

'''

''' layers = 10
    lr=0.01
    iterations = 25000
    rolling_avg = 1000 
    129/926
    
    '''
    
    
''' 
layers = 15
lr = 0.075
iteration = 25000
rolling avg = 1000
Correct:  145.0
Out of:  926'''


''' 
layers 15
iteration = 50000
rolling avg = 1000
Correct:  171.0
Out of:  926
0.18466522678185746
'''