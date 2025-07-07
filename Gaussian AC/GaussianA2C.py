import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import networkx as nx

import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

import matplotlib.pyplot as plt

from helper_functions import check_connectivity, create_state_graph


    
class TransformerPolicy(nn.Module):
    def __init__(self, 
                 input_size, 
                 action_dim, 
                 mask_action_space, 
                 mask_rate, 
                 lower_bound, 
                 upper_bound, 
                 nhead=4, 
                 num_encoder_layers=2, 
                 dim_feedforward=64, 
                 dropout=0.1):
        
        super(TransformerPolicy, self).__init__()

        # Transformer encoder definition
        self.embedding = nn.Linear(input_size, dim_feedforward)
        self.positional_encoding = PositionalEncoding(dim_feedforward, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # Actor's layers: mean and covariance
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.action_dim = action_dim
        self.mean_head = nn.Linear(dim_feedforward, action_dim)
        self.log_std_head = nn.Linear(dim_feedforward, action_dim)  # Log of standard deviations
#         self.cov_head = nn.Linear(dim_feedforward, action_dim * action_dim)  # For covariance matrix (flattened)

        self.mask_rate = mask_rate
        self.mask_heads = nn.ModuleList([nn.Linear(dim_feedforward, action_size) for action_size in mask_action_space])

        # Critic's layer
        self.value_head = nn.Linear(dim_feedforward, 1)

        # Action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x, iteration):
        x = self.embedding(x)  # Convert to embedding space
        x = self.positional_encoding(x)

        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, embedding_size)

        encoded_output = self.transformer_encoder(x)

        encoded_output = encoded_output[-1]  # Shape: (batch_size, dim_feedforward)

        # Actor: choose action means, log stds, and covariance matrix
        means = self.mean_head(encoded_output)
        means = torch.sigmoid(means) * (self.upper_bound - self.lower_bound) + self.lower_bound
        log_stds = self.log_std_head(encoded_output)
#         cov_flat = self.cov_head(encoded_output)
        # Reshape the covariance output to a square matrix
#         cov_matrix = cov_flat.view(-1, self.action_dim, self.action_dim)
#         cov_matrix = torch.matmul(cov_matrix, cov_matrix.transpose(1, 2)) + 1e-5 * torch.eye(self.action_dim).to(cov_matrix.device)  # Ensure positive semi-definiteness

        # Mask: choose which coefficients to mask
        if iteration % self.mask_rate == 0:
            self.mask_probs = [F.softmax(head(encoded_output), dim=-1) for head in self.mask_heads]

        # Critic: evaluate the value of being in the current state
        state_values = self.value_head(encoded_output)

        return means, log_stds, state_values, self.mask_probs
    
    
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding class to add positional information to the input embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix of size (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register pe as a buffer to avoid it being a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input tensor.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
    
    
    
    
    
def freeze_parameters(model, mask):
    # Mask for mean_head
    for i in range(len(mask)):
        if mask[i] == 0:  # Freeze if the value is 0
            for param in model.mean_head.parameters():
                param.requires_grad = False
        else:  # Keep trainable if the value is 1
            for param in model.mean_head.parameters():
                param.requires_grad = True

    # Mask for log_std_head
    for i in range(len(mask)):
        if mask[i] == 0:  # Freeze if the value is 0
            for param in model.log_std_head.parameters():
                param.requires_grad = False
        else:  # Keep trainable if the value is 1
            for param in model.log_std_head.parameters():
                param.requires_grad = True

    return model
    
    
def generate_mask(length, N):
    # Create an array of zeros
    mask = np.zeros(length, dtype=int)
    
    # Randomly choose N unique indices to set to 1
    indices = np.random.choice(length, size=N, replace=False)
    mask[indices] = 1
    mask = mask.tolist()
    return mask
    
    
    

def select_action_transformer(model, state, SavedAction, action_space_values, mask_range, iteration, testing):
    state = np.array(state)
    state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)

    means, log_stds, state_value, mask_probs = model(state, iteration)
    
    actions = []
    log_probs = []
    masks = []
    
    # Sample from the multivariate Gaussian
#     for mean, log_std in zip(means, log_stds):
    stds = torch.exp(log_stds)  # Standard deviations
#         mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_matrix)  # Multivariate Gaussian
    mvn = Normal(means, stds)
    action = mvn.sample()  # Sample from the distribution
    actions.append(action)  # Store the sampled action
    log_probs.append(mvn.log_prob(action))  # Store the log probability

    # Process masks
    for prob in mask_probs:
        prob = prob.squeeze(0)
        m = Categorical(prob)
        mask = m.sample()
        masks.append(mask)

    # Save to action buffer
    if not testing:
        model.saved_actions.append(SavedAction(log_probs, state_value, actions, masks))
    
    actions = [action.item() for action in action.squeeze()]
    masks = [mask.item() + mask_range * i for i, mask in enumerate(masks)]
    mask = [1 if i in masks else 0 for i in range(len(actions))]
    selected_action = [a * b for a, b in zip(actions, mask)]

    return selected_action #selected_action


def select_best_action_transformer(model, state, action_space_values, iteration):
    state = np.array(state)
    state = torch.from_numpy(state).float()
    state = state.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size), assuming batch_size=1 and sequence_length=1
    
    # Get means, log stds, and mask probabilities from the model
    means, log_stds, state_value, mask_probs = model(state, iteration)
    
    stds = torch.exp(log_stds)  # Standard deviations
    mvn = Normal(means, stds)
    print(means)
    # Get action probabilities and the sampled action
    action_probs = mvn.log_prob(means)  # Calculate log probabilities for the means
    action = means  # Directly use means for the action since we want the best (mean action)
    
    # Find the best action based on probabilities
    best_action_idx = torch.argmax(action_probs).item()
    best_action = action_space_values[best_action_idx]

    # Concatenate mask probabilities into a single tensor
    mask_probs_tensor = torch.cat(mask_probs, dim=-1)  # Assuming mask_probs are along the last dimension
    best_mask_idx = torch.argmax(mask_probs_tensor).item()
    best_mask = 1 if best_mask_idx == best_action_idx else 0  # Assuming a simple mask logic

    # Combine action and mask
    masked_action = best_action * best_mask  # Adjust based on your specific requirements
    print(best_mask)
    return best_action





def mask_action_probs(state, probs, action_space_values, node_num):
    new_probs = {}
    
    action_rounded = np.array(np.round(action), dtype=int)
    all_actions = [np.multiply(action_rounded[i], self.basis_move[i]) for i in range(len(action_rounded))]
    all_actions = np.stack(all_actions)
    self.action = np.sum(all_actions, 0)
    
    # Iterate over action_space_values to update new_probs
    for action in action_space_values:
        state_edges = state_edges_og.copy()
      
        if not bool(set(state_edges[action[0]]) & set(state_edges[action[1]])):
            e1 = state_edges[action[0]]
            e2 = state_edges[action[1]]

            state_edges.remove(e1)
            state_edges.remove(e2)

            new_e_1 = (e1[0], e2[1])
            new_e_2 = (e1[1], e2[0])

            if new_e_1[0] != new_e_1[1] and new_e_2[0] != new_e_2[1]:
                temp_edges = state_edges + [new_e_1, new_e_2]
                temp_g = nx.MultiGraph()
                temp_g.add_edges_from(temp_edges)
                if check_connectivity(temp_g, 1):
                    new_probs[action_space_values.index(action)] = probs[action_space_values.index(action)]
                else:
                    new_probs[action_space_values.index(action)] = 0
            else:
                new_e_1 = (e1[0], e2[0])
                new_e_2 = (e1[1], e2[1])
                temp_edges = state_edges + [new_e_1, new_e_2]
                temp_g = nx.MultiGraph()
                temp_g.add_edges_from(temp_edges)
                if check_connectivity(temp_g, 1):
                    new_probs[action_space_values.index(action)] = probs[action_space_values.index(action)]
                else:
                    new_probs[action_space_values.index(action)] = 0
        else:
            new_probs[action_space_values.index(action)] = 0
    
    # Create a list of probabilities based on new_probs with zeros for missing actions
    new_probs_list = [0] * (max(new_probs.keys()) + 1)

    for index, prob in new_probs.items():
        new_probs_list[index] = prob
        

    # Normalize the probabilities
    total = sum(list(new_probs.values()))
    if total > 0:
        normalized_list = [prob / total for prob in new_probs_list]
    else:
        normalized_list = new_probs_list  # In case the sum is zero (unlikely for probabilities)

    normalized_tensor = torch.tensor(normalized_list, dtype=torch.float32, device=probs.device)
    return normalized_tensor





def run_n_step_with_gae(model, n_step, gamma, lam, optimizer, scheduler_actor, scheduler_critic, done):
    """
    Training code for N-step updates with GAE (Generalized Advantage Estimation).
    """
    if len(model.rewards) < n_step and not done:
        # Wait until we have enough steps to compute n-step return
        return None, None
    
    
    saved_actions = model.saved_actions[:n_step]  # Take only the first N steps
    policy_losses = []  # List to save actor (policy) loss
    value_losses = []   # List to save critic (value) loss
    mask_losses = []    # List to save mask loss
    entropy_losses = [] # List to save entropy loss
    advantages = []     # List to store GAE advantages
    returns = []        # List to save the true values
    values = [value for _, value, probs, mask in saved_actions]
    R = values[-1] 
    
    # Calculate the GAE advantage for each step
    for t in reversed(range(len(model.rewards[:n_step]))):
        if t == len(model.rewards[:n_step]) - 1:
            next_value = values[t]  # Last step value estimate
        else:
            next_value = values[t + 1]
        
        # TD residual (δ_t)
        delta = model.rewards[t] + gamma * next_value.item() - values[t].item()

        # GAE advantage estimate
        if len(advantages) == 0:  # First step (most recent step)
            A_t = delta
        else:
            A_t = delta + gamma * lam * advantages[0]  # Recursively calculate GAE
        
        advantages.insert(0, A_t)  # Prepend to the list (reverse order)

    advantages = torch.tensor(advantages)
    eps = np.finfo(np.float32).eps.item()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)  # Normalize GAE

    # Calculate the discounted returns
#     for (log_probs, value, probs, mask_log_probs), R in zip(saved_actions, returns):
        
    for r in model.rewards[:n_step-1][::-1]: # should we start collecting the rewards 1 step before
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)  # Normalize returns
    
#     value_loss = torch.nn.MSELoss()
    value_loss = torch.nn.L1Loss()
    
    # Compute the policy loss and value loss for N-step actions
    for (log_probs, value, probs, mask_log_probs), A_t, R in zip(saved_actions, advantages, returns):
        
        # Calculate actor (policy) loss using GAE
        total_log_prob = sum(log_probs)  # Sum log probabilities for all action components
        policy_losses.append(-total_log_prob * A_t)

        # Calculate critic (value) loss using MS0
        value_losses.append(value_loss(value, torch.tensor([R]).float())**2) # missing the last V(S_{t+K})
        
        # Calculate mask loss using the same advantage
        total_mask_log_probs = sum(mask_log_probs) # Sum mask log probabilities for all action components
        mask_losses.append(-total_mask_log_probs * A_t)  # Align mask loss with policy loss
        
        entropy = -sum(torch.sum(F.softmax(prob, dim=-1) * torch.log(F.softmax(prob, dim=-1) + 1e-6)) for prob in log_probs) 
        entropy_losses.append(entropy)
        
    optimizer.zero_grad()

    # Sum up all the policy and value losses
    policy_loss = torch.stack(policy_losses).sum()/len(policy_losses)
    value_loss = torch.stack(value_losses).sum()/(2*len(value_losses))
    mask_loss = torch.stack(mask_losses).sum()/len(policy_losses)
    entropy_loss = torch.stack(entropy_losses).sum()
    
    total_loss = policy_loss + value_loss + mask_loss - 0.07 * entropy_loss
    
    # Backpropagate actor and critic losses independently
    total_loss.backward()

    # Perform optimizer step for both actor and critic
    optimizer.step()
    
    scheduler_actor.step()
    scheduler_critic.step()
    
#     decay_factor1 = lr_actor_gamma ** (1 + 0.0000000001 * training_period)
#     decay_factor2 = lr_critic_gamma ** (1 + 0.0000000001 * training_period)
    optimizer.param_groups[1]['lr'] *= scheduler_actor.get_last_lr()[0]
    optimizer.param_groups[2]['lr'] *= scheduler_critic.get_last_lr()[0]
    
    # Clear out the first N steps in the buffer (rewards and saved actions)
    del model.rewards[:n_step]
    del model.saved_actions[:n_step]
    
    return scheduler_actor.get_last_lr()[0], scheduler_critic.get_last_lr()[0]


def run_n_step_with_gae_cont(model, n_step, gamma, lam, optimizer, scheduler_actor, scheduler_critic, lr_actor_gamma, lr_critic_gamma, entropy_param, training_period, iteration, done):
    """
    Training code for N-step updates with GAE (Generalized Advantage Estimation).
    """
    if (iteration+1) % n_step != 0 and not done:
        # Wait until we have enough steps to compute n-step return
        return None, None, training_period
    
    training_period+=1
    R = 0
    saved_actions = model.saved_actions[:n_step]  # Take only the first N steps
    policy_losses = []  # List to save actor (policy) loss
    value_losses = []   # List to save critic (value) loss
    mask_losses = []    # List to save mask loss
    entropy_losses = [] # List to save entropy loss
    advantages = []     # List to store GAE advantages
    returns = []        # List to save the true values
    values = [value for _, value, probs, mask in saved_actions]

    # Calculate the GAE advantage for each step
    for t in reversed(range(len(model.rewards[:n_step]))):
        if t == len(model.rewards[:n_step]) - 1:
            next_value = values[t]  # Last step value estimate
        else:
            next_value = values[t + 1]
        
        # TD residual (δ_t)
        delta = model.rewards[t] + gamma * next_value.item() - values[t].item()
        # GAE advantage estimate
        if len(advantages) == 0:  # First step (most recent step)
            A_t = delta
        else:
            A_t = delta + gamma * lam * advantages[0]  # Recursively calculate GAE
        
        advantages.insert(0, A_t)  # Prepend to the list (reverse order)

    advantages = torch.tensor(advantages)
    eps = np.finfo(np.float32).eps.item()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)  # Normalize GAE

    # Calculate the discounted returns (not used in GAE but needed for value loss)
    for r in model.rewards[:n_step][::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)  # Normalize returns
    
    value_loss = torch.nn.MSELoss()
    
    # Compute the policy loss and value loss for N-step actions
    for (log_probs, value, probs, mask_log_probs), A_t, R in zip(saved_actions, advantages, returns):
        
        total_log_prob = sum(log_probs)  # Sum log probabilities for all action components
        
        # Calculate actor (policy) loss using GAE
        policy_losses.append(-total_log_prob * A_t)

        # Calculate critic (value) loss using MS0
        value_losses.append(value_loss(value, torch.tensor([R]).float()))
        
        # Calculate mask loss using the same advantage
        total_mask_log_probs = sum(mask_log_probs) # Sum mask log probabilities for all action components
        mask_losses.append(-total_mask_log_probs * A_t)  # Align mask loss with policy loss
        
        entropy = -sum(torch.sum(F.softmax(prob, dim=-1) * torch.log(F.softmax(prob, dim=-1) + 1e-6)) for prob in log_probs) 
        entropy_losses.append(entropy)
        
    optimizer.zero_grad()

    # Sum up all the policy and value losses
    policy_loss = torch.stack(policy_losses).sum() 
    value_loss = torch.stack(value_losses).sum()
    mask_loss = torch.stack(mask_losses).sum()
    entropy_loss = torch.stack(entropy_losses).sum()
    
    total_loss = policy_loss + value_loss + mask_loss + entropy_param * entropy_loss
    
    # Backpropagate actor and critic losses independently
    total_loss.backward()

    # Perform optimizer step for both actor and critic
    optimizer.step()

    # scheduler_actor.step()
    # scheduler_critic.step()
    
#     optimizer.param_groups[1]['lr'] *= 0.99#lr_actor_gamma#scheduler_actor.get_last_lr()[0]
#     optimizer.param_groups[2]['lr'] *= 0.95#lr_critic_gamma# scheduler_critic.get_last_lr()[0]
    decay_factor1 = lr_actor_gamma ** (1 + 0.0000000001 * training_period)
    decay_factor2 = lr_critic_gamma ** (1 + 0.0000000001 * training_period)
    optimizer.param_groups[1]['lr'] *= decay_factor1
    optimizer.param_groups[2]['lr'] *= decay_factor2

    # print("Actor LR: ", optimizer.param_groups[1]['lr'] )
    # print("Critic LR: ", optimizer.param_groups[2]['lr'] )

    
    
    
    # Clear out the first N steps in the buffer (rewards and saved actions)
    del model.rewards[:n_step]
    del model.saved_actions[:n_step]
    
    return optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'], training_period




def visualize_gaussian_distribution(model, state):
    # Prepare the state
    state = np.array(state)
    state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)  # (1, 1, input_size)

    # Get means and log stds from the model
    means, log_stds, _, _ = model(state, iteration=0)  # Replace with your iteration logic
    means = means.detach().numpy().squeeze()  # Convert to NumPy and remove extra dimensions
    stds = torch.exp(log_stds).detach().numpy().squeeze()  # Standard deviations

    # Create a range of values for the action
    x = np.linspace(means - 3 * stds, means + 3 * stds, 100)

    # Create a Gaussian distribution
    gaussian = Normal(torch.tensor(means), torch.tensor(stds))
    y = gaussian.log_prob(torch.tensor(x)).exp().detach().numpy()  # Probability density function

    # Plot the Gaussian distribution
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Gaussian Distribution', color='blue')
    plt.axvline(means, color='red', linestyle='--', label='Mean')
    plt.axvline(means + stds, color='green', linestyle='--', label='Mean + 1 Std Dev')
    plt.axvline(means - stds, color='green', linestyle='--', label='Mean - 1 Std Dev')
    plt.title('Gaussian Distribution of Actions')
    plt.xlabel('Action Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid()
    plt.show()