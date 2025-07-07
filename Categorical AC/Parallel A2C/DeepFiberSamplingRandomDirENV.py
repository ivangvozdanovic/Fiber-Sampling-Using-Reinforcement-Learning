import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
from operator import add
from gym import spaces, Env
import os
import math as m
import networkx as nx
import pickle
import sys
import logging

from multiprocessing import Queue, Process
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')


from helper_functions import reconnect_graph_generalized_version

def log_writer(log_queue):
    """ Continuously read log messages from the queue and print them """
    while True:
        try:
            message = log_queue.get(timeout=1)
            if message == "DONE":
                break
            print(message)
        except Exception as e:
            pass  # Ignore queue timeouts


class PolytopeENV(Env):
    def __init__(self, 
                 initial_states,
                 total_episodes, 
                 show_path_num, 
                 visited_states, 
                 basis_moves, 
                 node_num, 
                 P, 
                 lb,
                 log_queue=None):
        """
        Custom environment for Polytope problem, based on a multi-discrete action space.
        """
        super(PolytopeENV, self).__init__()

        self.log_queue = log_queue
        if log_queue:
            log_queue.put("Environment initialized.")
        else:
            print("Environment initialized.")
        
        # Environment settings
        self.initial_states = initial_states
        self.basis_move = basis_moves
        self.node_num = node_num
       
        self.P = P
        self.lb = lb
        self.visited_states = visited_states
        self.show_path_num = show_path_num
        self.total_episodes = total_episodes
        self.episode = -1

        self.random_direction = []
        self.recently_visited_states = []
        
        # Action space (assuming each action component can take multiple discrete values)
        num_action_components = len(basis_moves)  # Number of dimensions in the action space
        self.action_space = spaces.MultiDiscrete([len(move) for move in self.basis_move])  # Define the multi-discrete action space

        # Observation space (assuming the state is represented by an array of integers)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=self.initial_states[0].shape, dtype=np.int32)
        
        
        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        self._iteration = 0
        self._total_reward = 0
        self.path = 0
        self.episode += 1
        
        
        # Start from a random visited state
        state_indx = 0
        
        state_indx = 0
        if len(self.initial_states.keys()) > 1:
            state_indx = random.randint(0,len(self.initial_states.keys())-1)
        state = self.initial_states[state_indx]
        self.state = state
        self.random_direction = np.random.randint(2, size=len(self.initial_states[0]))
        
        if self.log_queue:
            self.log_queue.put(f"Environment reset: Starting episode {self.episode} with initial state {self.state}")
        else:
            print(f"Environment reset: Starting episode {self.episode} with initial state {self.state}")

        
        return self.state  # Return the initial state self.initial_states#

    def step(self, action):

        """Take a step in the environment."""
        action_rounded = np.array(np.round(action), dtype=int)
        all_actions = [np.multiply(action_rounded[i], self.basis_move[i]) for i in range(len(action_rounded))]
        all_actions = np.stack(all_actions)
        self.action = np.sum(all_actions, 0)
        

        if self._iteration % 20 == 0:
            self.random_direction = 10*np.random.randint(2, size=len(self.initial_states[0]))
            
        self._iteration += 1
        done = False
        found_solution = False
        info = {}

        # Update the state
        next_state = np.add(self.state, self.action)

        # Compute reward components
        reward_feasibility = 0
        reward_non_zero_action = 0
        reward_direction = 0
        
        if np.all(self.action == 0):
            logging.info("Action is a zero vector!")
            reward_non_zero_action = -10
            next_state = self.state
        else:
            reward_non_zero_action = 0

            if all(coord >= 0 for coord in next_state):

                # if next_state.tolist() in self.recently_visited_states:
                #     reward_feasibility = -10
                # else:
                #     self.recently_visited_states.append(next_state.tolist())
                #     if len(self.recently_visited_states) > 5:
                #         self.recently_visited_states.pop(0)
                reward_direction = np.dot(np.transpose(next_state), self.random_direction)
                
                if next_state.tolist() not in self.visited_states.tolist():  
                    logging.info("New state found!")
                    # reward_feasibility = 10
                    self.visited_states = np.concatenate((self.visited_states,[next_state]),axis=0)
              
                    
            else:
                for coord in next_state:
                    if coord < 0:
                        reward_feasibility += 10 * coord**2
                reward_feasibility = -np.sqrt(reward_feasibility)
                next_state = self.state

        reward = reward_feasibility + reward_non_zero_action + reward_direction
        self._total_reward += reward

        
        # Define a done condition (e.g., maximum iterations)
        if self._iteration >= self.show_path_num:  # You can define a suitable condition based on your problem
            logging.info(f'Episode: {self.episode} ||| Reward: {self._total_reward} ||| Discovered States: {len(self.visited_states)}')
#             done = True
            
        self.state = next_state

        return self.state, reward, done, info
    


   