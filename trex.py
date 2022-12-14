import gym
import pickle

from human import TerminalHuman
from keyboard_agent import *
from utils import *
from LearnAtariReward import *
import matplotlib.pyplot as plt
import torch.optim as optim

import os
import glob

# use cpsc672 conda environment

env = gym.make('LunarLander-v2')

demo_path = 'data/demonstrations'
demo_files = glob.glob(f"{demo_path}/gym-*.pickle")

demonstrations = [pickle.load(open(demo_name, 'rb'), encoding='latin1') for demo_name in demo_files] # load from files

# sorted best to worst demos
sorted_demonstrations = [(states, controls) for controls, states, reward in sorted(demonstrations, key=lambda pair: pair[2])]
sorted_rewards = [reward for controls, states, reward in sorted(demonstrations, key=lambda pair: pair[2])]


num_trajs =  0
num_snippets = 6000
min_snippet_length =  min(np.min([len(d[0]) for d in sorted_demonstrations]), 30) #min length of trajectory for training comparison
maximum_snippet_length = 100
max_snippet_length = min(np.min([len(d[0]) for d in sorted_demonstrations]), maximum_snippet_length)

# auto labelling, but this can be done with preference ranking
training_obs, training_labels = create_training_data(sorted_demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length)


testing_obs = training_obs[:1000]
training_obs = training_obs[1000:]

testing_labels = training_labels[:1000]
training_labels = training_labels[1000:]
len(testing_obs)
len(training_obs)

lr = 0.0005
weight_decay = 0.0
num_iter = 10 #num times through training data
l1_reg=0.0




reward_net = Net(3)


optimizer = optim.Adam(reward_net.parameters(),  lr=1e-4)#lr, weight_decay=weight_decay)

reward_model_path = 'reward_model/model0.pth'

learn_reward(reward_net, optimizer, training_obs, training_labels, testing_obs, testing_labels, num_iter, l1_reg, reward_model_path) #path is where to save the model
#save reward network
torch.save(reward_net.state_dict(), reward_model_path)


with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, np_features(traj[0])) for traj in sorted_demonstrations]
        # pred_returns = [predict_traj_return(reward_net, traj[0]) for traj in sorted_demonstrations]
for i, p in enumerate(pred_returns):
    print(i,sorted_rewards[i], p)
    
    
plt.scatter(sorted_rewards,pred_returns)
plt.savefig('trex_1.png')


num_queries = 10
timestep_length = 250

policies = np.random.choice([0, 1, 2, 3], size=(num_queries, timestep_length))
max_policy = max(policies, key= lambda x: predict_traj_return(reward_net, np_features(watch(env, x, render=True))))

'''
show max policy
'''
print('showing max policy')
watch(env, max_policy)

