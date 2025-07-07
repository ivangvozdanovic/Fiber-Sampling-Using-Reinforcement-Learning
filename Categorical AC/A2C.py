import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import networkx as nx
from itertools import product
from functools import reduce
import operator

import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


from helper_functions import check_connectivity, create_state_graph



class Policy(nn.Module):
   
    def __init__(self, feature_arch, input_size, output_size):
        super(Policy, self).__init__()
        feature_net = []
        for l in range(len(feature_arch)-1):
            feature_net.append(nn.Linear(feature_arch[l], feature_arch[l+1]))
            feature_net.append(nn.ReLU())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_net = nn.Sequential(*feature_net).to(device)
        
        # actor's layer
        self.action_head = nn.Linear(feature_arch[len(feature_arch)-1], output_size)
        
        # critic's layer
        self.value_head = nn.Linear(feature_arch[len(feature_arch)-1], 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x =  self.feature_net(x)
        
        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
    

    
    
class TransformerPolicy(nn.Module):
   
    def __init__(self, input_size, action_space, mask_action_space, mask_rate, nhead=4, num_encoder_layers=2, dim_feedforward=16, dropout=0.1):
        super(TransformerPolicy, self).__init__()
        
        # Transformer encoder definition
        self.embedding = nn.Linear(input_size, dim_feedforward)  # Convert binary input to embedding dimension
        self.positional_encoding = PositionalEncoding(dim_feedforward, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # actor's layer
        self.action_heads = nn.ModuleList([nn.Linear(dim_feedforward, action_size) for action_size in action_space])
    
        self.mask_rate = mask_rate
        self.mask_probs = None
        self.mask_heads = nn.ModuleList([nn.Linear(dim_feedforward, action_size) for action_size in mask_action_space])
        
        # critic's layer
        self.value_head = nn.Linear(dim_feedforward, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []


    def forward(self, x, iteration):
        """
        Forward pass for both actor and critic using Transformer encoder for feature extraction.
        """
        # Input x should be of shape (batch_size, sequence_length, input_size)
        x = self.embedding(x)  # Convert to embedding space
        x = self.positional_encoding(x)

        # Transformer expects input in (sequence_length, batch_size, embedding_size)
        x = x.permute(1, 0, 2)  # From (batch_size, sequence_length, embedding_size) to (sequence_length, batch_size, embedding_size)

        # Pass through the Transformer encoder
        encoded_output = self.transformer_encoder(x)

        # Take the output of the last token (sequence_length - 1) for final decision making
        encoded_output = encoded_output[-1]  # Shape: (batch_size, dim_feedforward)

        # actor: choose action probabilities from the encoded state
        action_probs = [F.softmax(head(encoded_output), dim=-1) for head in self.action_heads]

        # mask: choose which coefficients to mask
        if iteration % self.mask_rate == 0:
            self.mask_probs = [F.softmax(head(encoded_output), dim=-1) for head in self.mask_heads]
    
        # critic: evaluate the value of being in the current state
        state_values = self.value_head(encoded_output)

        return action_probs, state_values, self.mask_probs
    
    
    
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
    for i, layer in enumerate(model.action_heads):
        if mask[i] == 0:  # Freeze if the value is 0
            for param in layer.parameters():
                param.requires_grad = False
        else:  # Keep trainable if the value is 1
            for param in layer.parameters():
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
    
    
 
def select_action(model, state, SavedAction, action_space_values, testing):
    state = np.array(state)
    state = torch.from_numpy(state).float()
    action_probs, state_value = model(state)
    
    actions = []
    log_probs = []

    for prob in action_probs:
        m = Categorical(prob)
        action = m.sample()  # Sample from the categorical distribution
        actions.append(action)  # Store the sampled action
        log_probs.append(m.log_prob(action))  # Store the log probability
        
    
    
    # save to action buffer
    model.saved_actions.append(SavedAction(log_probs, state_value))

    actions = [action.item() for action in actions]
    selected_action = [action_space_values[i][action] for i, action in enumerate(actions)]
    
    return selected_action


def select_action_transformer(model, state, SavedAction, action_space_values, mask_range, iteration, use_mask, testing):
    # Convert state to numpy array and reshape it as (sequence_length, input_size)
    state = np.array(state)
    state = torch.from_numpy(state).float()

    # Reshape state for Transformer input: 
    # If your state is a single 0/1 vector, treat it as a sequence of length 1
    state = state.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size), assuming batch_size=1 and sequence_length=1

    # Get the action probabilities and state value from the model (Transformer-based Policy)
    action_probs, state_value, mask_probs = model(state, iteration)
   
    actions = []
    log_probs = []
    probs = []
    mask_log_probs = []
    masks = []

    
    for prob in action_probs:
        prob = prob.squeeze(0)
        probs.append(prob)
        m = Categorical(prob)
        action = m.sample()  # Sample from the categorical distribution
        actions.append(action)  # Store the sampled action
        log_probs.append(m.log_prob(action))  # Store the log probability
    
        
    # save to action buffer
    model.saved_actions.append(SavedAction(log_probs, state_value, probs, mask_log_probs))
    actions = [action.item() for action in actions]
    selected_action = [action_space_values[i][action] for i, action in enumerate(actions)]
    mask = [1 if i in masks else 0 for i in range(len(selected_action))]
    if use_mask == True:
        
        for prob in mask_probs:
            prob = prob.squeeze(0)
            m = Categorical(prob)
            mask = m.sample()
            masks.append(mask)
        mask_log_probs.append(m.log_prob(mask))
        masks = [mask.item()+mask_range*i for i, mask in enumerate(masks)]
        selected_action = [a * b for a, b in zip(selected_action, mask)]
        
#     selected_action = selected_action[0].tolist()
    return selected_action


def construct_stochastic_policy(model, state, SavedAction, action_space_values, mask_range, iteration, use_mask, prob_treshold):
    # Convert state to numpy array and reshape it as (sequence_length, input_size)
    state = np.array(state)
    state = torch.from_numpy(state).float()

    # Reshape state for Transformer input: 
    # If your state is a single 0/1 vector, treat it as a sequence of length 1
    state = state.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size), assuming batch_size=1 and sequence_length=1

    # Get the action probabilities and state value from the model (Transformer-based Policy)
    action_probs, state_value, mask_probs = model(state, iteration)
   
    actions = []
    log_probs = []
    probs = []
    mask_log_probs = []
    masks = []

    action_probs = [prob.detach() for prob in action_probs]
    print(action_probs)
    # ordered_actions = get_ordered_actions(action_probs, prob_treshold)
    ordered_actions = get_sorted_combinations(action_probs, prob_treshold)

    
    for prob in mask_probs:
        prob = prob.squeeze(0)
        m = Categorical(prob)
        mask = m.sample()
        # masks.append(mask)
        # mask_log_probs.append(m.log_prob(mask))
        
    # save to action buffer
    # model.saved_actions.append(SavedAction(log_probs, state_value, probs, mask_log_probs))

    actions = [action.item() for action in actions]
    print(actions)
    masks = [mask.item()+mask_range*i for i, mask in enumerate(masks)]

    selected_actions = []
    action_probabilities = []
    for a in ordered_actions:
        selected_action = [action_space_values[i][action] for i, action in enumerate(list(a[0]))]
        action_probabilities.append(a[1])
        selected_actions.append(selected_action)
    print("Selected actions: ", selected_actions)
    mask = [1 if i in masks else 0 for i in range(len(selected_actions))]
    if use_mask == True:
        selected_actions = [a * b for a, b in zip(selected_actions, mask)]

    return selected_actions, action_probabilities


# def get_ordered_actions(probabilities):
#     # Initialize a list to store lists of actions ordered by probability ranking
#     ordered_actions = []
    
#     # Process each probability tensor
#     for prob in probabilities:
#         # Ensure the tensor is 1D by squeezing any extra dimensions
#         prob = prob.squeeze()
        
#         # Get the sorted indices in descending order of probability values
#         sorted_indices = torch.argsort(prob, descending=True).tolist()
        
#         # Append sorted indices as a list (highest to lowest) to ordered_actions
#         ordered_actions.append(sorted_indices)
        
#     # Transpose the list to get actions based on their rank across all tensors
#     ordered_actions_by_rank = list(map(list, zip(*ordered_actions)))
    
#     return ordered_actions_by_rank


def get_sorted_combinations(probabilities, threshold=0.0):
    # Filter out low-probability elements in each row based on the threshold
    filtered_probs = []
    for prob in probabilities:
        prob = prob.squeeze()  # Ensure it's 1D
        mask = prob >= threshold  # Apply threshold
        filtered_probs.append([(i, p.item()) for i, p in enumerate(prob) if mask[i]]) 

    # Generate all possible combinations of actions across rows
    all_combinations = list(product(*filtered_probs))

    # Calculate average probabilities for each combination
    combination_probs = []
    for combination in all_combinations:
        indices, probs = zip(*combination)  # Separate actions and their probabilities
        avg_prob = sum(probs)# / len(probs)  # Compute the average probability
        #avg_prob = reduce(operator.mul, probs)  # Compute the joint probabilities.
        # if avg_prob > threshold:
        combination_probs.append((indices, avg_prob))

    print(combination_probs)
    # Sort combinations by average probability in descending order
    sorted_combinations = sorted(combination_probs, key=lambda x: x[1], reverse=True)
    print(sorted_combinations)
    print("######")
    return sorted_combinations


def get_ordered_actions(probabilities, threshold=0.0):
    # Initialize a list to store lists of actions ordered by probability ranking
    ordered_actions = []
    
    # Process each probability tensor
    for prob in probabilities:
        # Ensure the tensor is 1D by squeezing any extra dimensions
        prob = prob.squeeze()
        
        # Apply threshold to filter probabilities
        mask = prob >= threshold
        filtered_prob = prob[mask]
        filtered_indices = torch.arange(len(prob))[mask]  # Get the indices of values above threshold
        
        # Sort the remaining probabilities and their indices in descending order
        sorted_indices = filtered_indices[torch.argsort(filtered_prob, descending=True)].tolist()
        
        # Append sorted indices as a list (highest to lowest) to ordered_actions
        ordered_actions.append(sorted_indices)
        
    # Transpose the list to get actions based on their rank across all tensors
    ordered_actions_by_rank = list(map(list, zip(*ordered_actions))) if ordered_actions else []
    
    return ordered_actions_by_rank


def select_best_action_transformer(model, state, action_space_values):
    state = np.array(state)
    state = torch.from_numpy(state).float()
    state = state.unsqueeze(0).unsqueeze(0)  # (1, 1, input_size), assuming batch_size=1 and sequence_length=1
    probs, state_value = model(state)
    action_probs = probs.squeeze(0) 
    print(action_probs)
    action = np.argmax(probs.detach().numpy())
    return action_space_values[action.item()]


def select_best_action(model, state, action_space_values):
    state = np.array(state)
    state = torch.from_numpy(state).float()
    action_probs, state_value = model(state)
    
    actions = []
    for prob in action_probs:
        action = np.argmax(prob.detach().numpy())
        actions.append(action)
        
    actions = [action.item() for action in actions]
    selected_action = [action_space_values[i][action] for i, action in enumerate(actions)]
    
    return selected_action




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





def run_n_step_with_gae(model, n_step, gamma, lam, optimizer, scheduler_actor, scheduler_critic, lr_actor_gamma, lr_critic_gamma, entropy_param, training_period, iteration, done):
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