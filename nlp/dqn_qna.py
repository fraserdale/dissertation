from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
import tqdm



rewards = []

def eval_model(env, model, pool):
    print("=======================")
    correctly_answered = 0.0
    for sample, index in tqdm.tqdm(pool, desc="Evaluating"):
        obs = env.reset(sample)
        state = None
        done = False
        
        while not done:
            action, state = model.predict(obs)
            ''' if correctly_answered < 5:
                print(action) '''
            obs, reward, done, info = env.step(action)

        if info["selected_choice"] == sample.answer:
            correctly_answered += 1
            
    rewards.append(correctly_answered/len(pool))

    return correctly_answered/len(pool)


# data pool
data_pool = QASC.prepare(split="train")
val_pool = QASC.prepare(split="val")

# featurizer
featurizer = InformedFeaturizer()




from nlp_gym.envs.common.observation import BaseObservation
from nlp_gym.envs.common.reward import RewardFunction

class MyRewardFunction(RewardFunction):
    def __call__(self, observation: BaseObservation, action: str, target: str) -> float:

        current_time_step = observation.get_current_time_step()
        total_time_steps = observation.get_total_steps()
        selected_choice = observation.get_last_choice()
        
        # if current action is ANSWER or ran out of input, then check the current choice and produce terminal reward
        if action == "ANSWER" or current_time_step == total_time_steps - 1:
            reward = 1.0 if selected_choice == target else 0.0
        elif action == "CONTINUE" and selected_choice != target:
            reward = 1.0
        else:
            reward = 0.0
            
        return reward

# seq tag env
env = QAEnv(observation_featurizer=featurizer)
for sample, weight in data_pool:
    env.add_sample(sample, weight)

# check the environment
check_env(env, warn=True)

# train a MLP Policy
model = DQN(env=env, policy=DQNPolicy, gamma=0.99, batch_size=64, learning_rate=1e-3,
            double_q=False, exploration_fraction=0.1,
            prioritized_replay=True, policy_kwargs={"layers": [64,64]},
            verbose=1)
for i in range(int(50)):
    model.learn(total_timesteps=int(1e+3), reset_num_timesteps=False)
    print(eval_model(env, model, val_pool))
    
print(rewards)

import pickle
with open('rewards.pkl', 'wb') as f:
    pickle.dump(rewards, f)
    