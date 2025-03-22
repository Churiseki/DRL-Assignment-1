# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

Q_FILE = "q_table.pkl"
A_FILE = "last_action.pkl"


if os.path.exists(Q_FILE):
    with open(Q_FILE, "rb") as f:
        Q_table = pickle.load(f)
else:
    Q_table = {}

def softmax(x):
    """✅ Compute softmax values for an array."""
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()

def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    if os.path.exists(A_FILE):
        with open(A_FILE, "rb") as f1:
            last_action = pickle.load(f1)
    else:
        last_action = 5


    obs_key = (last_action, obs[10], obs[11], obs[12], obs[13], obs[14], obs[15])
    if obs_key not in Q_table:
        Q_table[obs_key] = np.random.uniform(-1, 1, 6).tolist()
    prob = torch.tensor(softmax(Q_table[obs_key]), dtype=torch.float32)
    action = int(torch.multinomial(prob, num_samples=1))
    
    # 更新 Q-table 並存回檔案
    with open(A_FILE, "wb") as f1:
        pickle.dump(action, f1)
    return action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

