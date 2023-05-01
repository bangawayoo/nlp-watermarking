from transformers import AutoTokenizer, AutoModelForMaskedLM

INFILL_TOKENIZER = AutoTokenizer.from_pretrained('bert-base-cased')
INFILL_MODEL = AutoModelForMaskedLM.from_pretrained("bert-base-cased")


# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def get_probs(self, state):
        """
        states: tensor of shape (bs, seq, hidden_size)

        flatten the states and make sure padded tokens get zero prob
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        return m
        # action = m.sample()
        # return action.item(), m.log_prob(action)




"""
Get token representation using LM. Output: tensor of shape (bs, seq, hidden size)
 - clean
 - corrupted

RL:
 - environment: token representations, selected token
 - at each time step, choose tokens 
 - max time steps: number of bits to represent for each sample  

Forward Policy Network
 - Output: probs, action, m.log_prob 

Reward 
 1. Robustness against infill
 2. Robustness against corruption  
 3. Semantic (similarity with original, fluency)
"""
from collections import deque

class Agent:
    def __init__(self, rewards):
        self.rewards = rewards


    def act(self, clean, corrupted, max_time, optimizer, policy):
        scores_deque = deque(maxlen=100)
        scores = []
        gamma = 0.9

        cl_prob = policy.forward(clean)
        corr_prob = policy.forward(corrupted)
        for idx, (cl, corr, max_t) in enumerate(zip(cl_prob, corr_prob, max_time)):
            t = 0
            R = []
            L_prob = []
            done = False
            while t < max_t or done:
                clean_dist = Categorical(cl)
                clean_action = clean_dist.sample()
                cl[clean_action] = 0

                corr_dist = Categorical(corr)
                corr_action = corr_dist.sample()
                cl[corr_action] = 0

                R.append(self.rewards(clean_dist, corr_dist))
                L_prob.append(clean_dist.log_prob(clean_action))
                t += 1

            scores_deque.append(sum(R))
            scores.append(sum(R))

            returns = deque(maxlen=max_t)
            n_steps = len(R)

            for t in range(n_steps)[::-1]:
                disc_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(gamma * disc_return_t + R[t])


            ## standardization of the returns is employed to make training more stable
            eps = np.finfo(np.float32).eps.item()
            ## eps is the smallest representable float, which is
            # added to the standard deviation of the returns to avoid numerical instabilities
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            policy_loss = []
            for log_prob, disc_return in zip(L_prob, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
        return scores