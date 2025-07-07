import numpy as np
import os
import networkx as nx
import itertools
from collections import deque
import time as Time
import torch
from collections import namedtuple
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


from DeepFiberSamplingENV import PolytopeENV as Env 

from reward_functions import reward_cost, calculate_reward1, calculate_reward2, calculate_reward3

from helper_functions import create_fiber_sampling_erdos_renyi_graph, \
                             extract_distance_matrix, \
                             create_real_data_graph, \
                             create_real_data_initial_sol,\
                             moving_average, \
                             create_state_graph, \
                             permute_moves




from A2C import Policy, \
                TransformerPolicy, \
                select_action, \
                select_action_transformer, \
                run_n_step_with_gae, \
                select_best_action, \
                select_best_action_transformer, \
                freeze_parameters, \
                generate_mask,\
                construct_stochastic_policy


num_agents = 2

node_num = 5
num_episodes = 300
lb = -1
ub = 2
mask_size = 5
mask_rate = 50
mask_action_size = [5//mask_size for i in range(mask_size)]
mask_action_size[-1] += 5%mask_size # add the remained if not divisible.
mask_range = mask_action_size[0]
action_space_values = [ [i+lb for i in range(ub-lb)] for j in range(5) ]
action_space_size = [ub-lb for i in range(5)]
initial_states = {0: np.array([0, 1, 0, 1, 1, 1, 1, 0, 0, 1])}
visited_states = [np.array(initial_states[0])]
visited_states = np.stack(visited_states)

policies = {}

for a_num in range(num_agents):
    feature_net_arch = [len(initial_states[0]), 56, 28, 12, 28]
    # model = Policy(feature_net_arch, len(initial_states[0]), len(action_space_values))
    model = TransformerPolicy(len(initial_states[0]), action_space_size, mask_action_size, mask_rate)
    model.load_state_dict(torch.load('Models/Agent_'+str(a_num)+'/policy_model_Node#_' + str(node_num) + '_EP_' + str(num_episodes) + '.pth')) #+ str(node_num)
    model.eval()
    policies[a_num] = model

for a_num in range(num_agents):
        


        print("########################################################################")
        print("########################################################################")
        print("########################################################################")
        print("########################################################################")
        print("########################################################################")
        print(f'For Agent {a_num} simulate a Markov Chain using the optimal policy:')
        
        # Convert dictionary values to a list of arrays
        # visited_states = [np.array(initial_states[0])]
        visited_states = np.stack(visited_states)
    
        #Initialize the environment.
        env = Env(initial_states, # initial_state
                 1, # total_episodes
                 50, # show_path_num
                 visited_states,  # visited_states
                 available_actions, # basis_moves
                 node_num, # node_num
                 0, # P
                 lb, #lb
                 )
    
        # reset environment and episode reward
        state = env.reset()
     
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(20):

            policy_num = np.random.randint(0,len(list(policies.keys())))
            print("Pick Agent: ", policy_num)
            model = policies[policy_num]
            action_coeffs = select_best_action_transformer(model, state, action_space_values, t)
            # action = select_action_transformer(model, state, SavedAction, action_space_values, mask_range, t, None, True)
            # actions, action_probabilities = construct_stochastic_policy(model, state, SavedAction, action_space_values, mask_range, t, False, 0.01)
            # action_choices = []
            # for a in action_coeffs:
            action_rounded = np.array(np.round(action_coeffs), dtype=int)
            all_actions = [np.multiply(action_rounded[i], available_actions[i]) for i in range(len(action_rounded))]
            all_actions = np.stack(all_actions)
            action = np.sum(all_actions, 0)
            
                
            # action_choices, filtered_actions_probs = filter_actions(state, action_choices, action_probabilities)
            # if len(action_choices) > 1:
            #     random_action = np.random.randint(0,len(action_choices))
            # else:
            #     random_action = 0
            # print(random_action)
            # action = action_choices[random_action]
            # print("Action: ",action, " with probability ", filtered_actions_probs[random_action])
            print("Action: ", action)
            next_state = np.add(state, action)
            print("Next state: ", next_state)
            if all(coord >= 0 for coord in next_state):
                if next_state.tolist() not in visited_states.tolist():
                    visited_states = np.concatenate((visited_states,[next_state]),axis=0)
                state = next_state
            print("#####################")
        
        print(f'We discovered {visited_states.shape[0]} unique states using the optimal stochastic policy')