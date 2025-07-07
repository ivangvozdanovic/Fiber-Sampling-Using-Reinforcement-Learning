
'''
    TODO: 
        After training the optimal policy, sample from it and record the action distribution and record the number of feasible moves vs. unfeasible moves. 
        See whether this optimal distribution is indeed optimal and makes erros with very small probability. 

        See whether there is some connection between the probability ditribution and the Markov/Grobner basis and algebraic theory.

'''






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


from DeepFiberSamplingRandomDirENV import PolytopeENV as Env 

from reward_functions import reward_cost, calculate_reward1, calculate_reward2, calculate_reward3

from helper_functions import create_fiber_sampling_erdos_renyi_graph, \
                             extract_distance_matrix, \
                             create_real_data_graph, \
                             create_real_data_initial_sol,\
                             moving_average, \
                             create_state_graph, \
                             permute_moves


# path_initial = os.getcwd() + os.sep + 'Real Data' + os.sep + 'LargestComponent' + os.sep + 'largeComponentMtx.txt'
path_initial = os.getcwd() + os.sep + 'Real Data' + os.sep + 'MediumComponent' + os.sep + 'nextComponentMtx.txt'

initial_states = {} # dictionary holding the initial states.
patches = 1
node_num = 5
p = 0.6
graph_num = 1

#Pick the file to the problem:
file = 'A2C_Fiber_Sampling_'


available_actions, initial_states = create_fiber_sampling_erdos_renyi_graph(file, initial_states, node_num, p, graph_num)
# initial_states, available_actions, node_num = create_real_data_graph(path_initial) # works for smaller problems where we can compute lattice.


save_data = True  # save Q table data and cost vector data.
save_plots = False  # save the plots
save_data_rate = 20



# Example usage for running episodes
num_episodes = 2000
max_path_length = 30



n_step = max_path_length
sheduler_lr_update = 10 # every 10 trainig periods we modify the step size
gamma = 0.9
lam = 0.5
discount_factor = gamma
entropy_param = 0.05

actor_target_lr = 0.0001
critic_target_lr = 0.00001  # Set a lower target for faster convergence

actor_lr = 0.0006
critic_lr = 0.0006
step_size = num_episodes//sheduler_lr_update
lr_actor_gamma = (actor_target_lr / actor_lr) ** (1 / sheduler_lr_update)
lr_critic_gamma = (critic_target_lr / critic_lr) ** (1 / sheduler_lr_update)
print(step_size)
print(lr_actor_gamma)
print(lr_critic_gamma)



lb = -1
ub = 2


use_mask = False
mask_size = 5
mask_rate = 50
mask_action_size = [len(available_actions)//mask_size for i in range(mask_size)]
mask_action_size[-1] += len(available_actions)%mask_size # add the remained if not divisible.
mask_range = mask_action_size[0]
print(mask_action_size)
action_space_values = [ [i+lb for i in range(ub-lb)] for j in range(len(available_actions)) ]
action_space_size = [ub-lb for i in range(len(available_actions))]
print(len(action_space_size))



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


feature_net_arch = [len(initial_states[0]), 56, 28, 12, 28]
# model = Policy(feature_net_arch, len(initial_states[0]), len(action_space_values))
model = TransformerPolicy(len(initial_states[0]), action_space_size, mask_action_size, mask_rate)
print(model)

optimizer = torch.optim.Adam([
    {'params': model.transformer_encoder.parameters()},   # Shared feature extractor
    {'params': model.action_heads.parameters(), 'lr': actor_lr},  # Actor-specific parameters
    {'params': model.value_head.parameters(), 'lr': critic_lr} # Critic-specific parameters
])

scheduler_actor = StepLR(optimizer, step_size=step_size, gamma=lr_actor_gamma)
scheduler_critic = StepLR(optimizer, step_size=step_size, gamma=lr_critic_gamma)


# actor_params = list(model.action_head.parameters()) + list(model.feature_net.parameters())  # Actor network params
# critic_params = list(model.value_head.parameters())  # Critic network params
# actor_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)  # Learning rate for actor
# critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)  # Learning rate for critic
eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_probs', 'value', 'probs', 'mask_log_probs'])




def run_episode(env_params, model_class, model_args, optimizer_args, scheduler_args, agent_id, num_episodes, save_data_rate, result_queue):
    """
    Function to run multiple episodes for a single agent in a separate process.
    Each agent has its own model, and the model is saved independently.
    """

    # Initialize the environment
    env = Env(*env_params)


    
    # Initialize a separate model instance for this agent
    model = model_class(**model_args)

    # Initialize optimizer and schedulers
    optimizer = torch.optim.Adam([
        {'params': model.transformer_encoder.parameters()},   # Shared feature extractor
        {'params': model.action_heads.parameters(), 'lr': optimizer_args['actor_lr']},  # Actor-specific parameters
        {'params': model.value_head.parameters(), 'lr': optimizer_args['critic_lr']}  # Critic-specific parameters
    ])
    scheduler_actor = StepLR(optimizer, **scheduler_args['actor'])
    scheduler_critic = StepLR(optimizer, **scheduler_args['critic'])

    # Initialize tracking variables
    episode_reward_list = []
    cumm_running_reward = 0
    cumm_running_reward_list = []
    actor_lr_list = []
    critic_lr_list = []
    robins_monro_condition = []
    iteration = 0

    # Run multiple episodes
    for i_episode in range(num_episodes):
        # Reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        for t in range(max_path_length):
            # Select action from policy
            action = select_action_transformer(model, state, SavedAction, action_space_values, mask_range, iteration, use_mask, False)

            # Take the action
            state, reward, done, _ = env.step(action)
            
            model.rewards.append(reward)
            ep_reward += (discount_factor**t) * reward
            cumm_running_reward += reward
            cumm_running_reward_list.append(reward)

            # Perform backpropagation step
            actor_lr, critic_lr = run_n_step_with_gae(
                model, n_step, gamma, lam, optimizer, scheduler_actor, scheduler_critic, lr_actor_gamma, lr_critic_gamma, entropy_param, done
            )
            
            if actor_lr is not None and critic_lr is not None:
                actor_lr_list.append(actor_lr)
                critic_lr_list.append(critic_lr)
                robins_monro_condition.append(critic_lr / actor_lr)

            if done:
                break

            iteration += 1

        # Save rewards and other metrics
        episode_reward_list.append(ep_reward)

        # Save the model periodically for this agent
        if (i_episode + 1) % save_data_rate == 0:
            os.makedirs(f'Models Random Dir/Agent_{agent_id}', exist_ok=True)
            torch.save(model.state_dict(), f'Models Random Dir/Agent_{agent_id}/policy_model_Node#_{node_num}_EP_{num_episodes}.pth') # + str(node_num) + "_EP_" + str(num_episodes)
            print(f'Agent {agent_id}: Saved model at episode {i_episode + 1}.')

    # Send results to the main process
    result_queue.put({
        'agent_id': agent_id,
        'episode_reward_list': episode_reward_list,
        'cumm_running_reward_list': cumm_running_reward_list,
        'actor_lr_list': actor_lr_list,
        'critic_lr_list': critic_lr_list,
        'robins_monro_condition': robins_monro_condition
    })



def run_independent_agents(env_params, model_class, model_args, optimizer_args, scheduler_args, num_agents, num_episodes, save_data_rate):
    
    start_time = Time.time()
    """
    Runs multiple independent agents in parallel, each with its own policy model.
    """
    mp.get_context("spawn")
    result_queue = mp.Queue()

    # Start each agent in its own process
    processes = []
    for agent_id in range(num_agents):
        p = mp.Process(target=run_episode, args=(env_params, model_class, model_args, optimizer_args, scheduler_args, agent_id, num_episodes, save_data_rate, result_queue))
        p.start()
        processes.append(p)
        print(f'Agent id {agent_id}')

    # Collect results from each agent
    all_agent_results = {}
    for _ in range(num_agents):
        result = result_queue.get()
        agent_id = result['agent_id']
        all_agent_results[agent_id] = result
        # print(result)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()

    end_time = Time.time()
    print(f'It took {(end_time-start_time)/60} minutes to run {num_episodes} episodes.')
    
    return all_agent_results





if __name__ == "__main__":
    # Define environment and model parameters
    visited_states = [np.array(initial_states[0])]
    visited_states = np.stack(visited_states)
    env_params = (
        initial_states, num_episodes, max_path_length, visited_states, available_actions, node_num, 0, lb
    )
    
    
    model_class = TransformerPolicy  # Class, not an instance
    model_args = {
        "input_size": len(initial_states[0]),
        "action_space": action_space_size,
        "mask_action_space": mask_action_size,
        "mask_rate": mask_rate
    }
    
    # Optimizer and scheduler parameters
    optimizer_args = {
        'actor_lr': actor_lr,
        'critic_lr': critic_lr
    }
    scheduler_args = {
        'actor': {'step_size': step_size, 'gamma': lr_actor_gamma},
        'critic': {'step_size': step_size, 'gamma': lr_critic_gamma}
    }
    
    # Run N independent agents
    num_agents = 1
    results = run_independent_agents(env_params, model_class, model_args, optimizer_args, scheduler_args, num_agents, num_episodes, save_data_rate)
    # print("Training results:", results)

    policies = {}

    for a_num in range(num_agents):
        feature_net_arch = [len(initial_states[0]), 56, 28, 12, 28]
        # model = Policy(feature_net_arch, len(initial_states[0]), len(action_space_values))
        model = TransformerPolicy(len(initial_states[0]), action_space_size, mask_action_size, mask_rate)
        model.load_state_dict(torch.load('Models Random Dir/Agent_'+str(a_num)+'/policy_model_Node#_' + str(node_num) + '_EP_' + str(num_episodes) + '.pth')) #+ str(node_num)
        model.eval()
        policies[a_num] = model



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
    for t in range(100):
        
        policy_num = np.random.randint(0,len(list(policies.keys())))
        print("Pick Agent: ", policy_num)
        model = policies[policy_num]
        action_coeffs = select_best_action_transformer(model, state, action_space_values, mask_range, t, use_mask)
        # action = select_action_transformer(model, state, SavedAction, action_space_values, mask_range, t, None, True)
        # actions, action_probabilities = construct_stochastic_policy(model, state, SavedAction, action_space_values, mask_range, t, False, 0.01)
     
        action_rounded = np.array(np.round(action_coeffs), dtype=int)
        all_actions = [np.multiply(action_rounded[i], available_actions[i]) for i in range(len(action_rounded))]
        all_actions = np.stack(all_actions)
        action = np.sum(all_actions, 0)
        
    
        print("Action: ", action)
        next_state = np.add(state, action)
        print("Next state: ", next_state)
        if all(coord >= 0 for coord in next_state):
            if next_state.tolist() not in visited_states.tolist():
                visited_states = np.concatenate((visited_states,[next_state]),axis=0)
            state = next_state
        print("#####################")
    
    print(f'We discovered {visited_states.shape[0]} unique states using the optimal stochastic policy')

    
    for a_num in range(num_agents):
        
        # cumm_running_reward_list = results[a_num]['cumm_running_reward_list']
        episode_reward_list = results[a_num]['episode_reward_list']
        # # loss_list = results[a_num]['loss_list']
        # actor_lr_list = results[a_num]['actor_lr_list']
        # critic_lr_list = results[a_num]['critic_lr_list']
        # robins_monro_condition = results[a_num]['robins_monro_condition']
        
        # cum_reward_x_axis = [i for i in range(len(cumm_running_reward_list))]
        # ep_reward_x_axis = [i for i in range(len(episode_reward_list))]
        # # loss_x_axis = [i for i in range(len(loss_list))]
        # actor_lr_x_axis = [i for i in range(len(actor_lr_list))]
        # critic_lr_x_axis = [i for i in range(len(critic_lr_list))]
        
        smoothed_rewards = moving_average(episode_reward_list, 10)
        smoothed_rewards_x_axis = [i for i in range(len(smoothed_rewards))]
        
        plt.plot(smoothed_rewards_x_axis, smoothed_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Moving Average")
        plt.show()
        
        # plt.plot(ep_reward_x_axis, episode_reward_list)
        # plt.xlabel("Episodes")
        # plt.ylabel("Episodic Rewards")
        # plt.show()
        # plt.plot(cum_reward_x_axis, cumm_running_reward_list)
        # plt.xlabel("Episodes")
        # plt.ylabel("Cumulative Reward")
        # plt.show()
        # plt.plot(actor_lr_x_axis, actor_lr_list)
        # plt.title("Actor loss")
        # plt.show()
        # plt.plot(critic_lr_x_axis, critic_lr_list)
        # plt.title("Critic loss")
        # plt.show()
        
        
        # plt.plot(critic_lr_x_axis,robins_monro_condition)
        # # plt.xticks(ticks=custom_x_values, labels=custom_x_labels, rotation=45, ha='right', size='small')
        # plt.grid(visible=True)
        # plt.title("Robins-Monro convergence condition")
        # plt.ylabel("(Actor LR)/(Critic LR) ")
        # plt.xlabel("Episodes")
        # plt.show()


        # feature_net_arch = [len(initial_states[0]), 56, 28, 12, 28]
        # # model = Policy(feature_net_arch, len(initial_states[0]), len(action_space_values))
        # model = TransformerPolicy(len(initial_states[0]), action_space_size, mask_action_size, mask_rate)
        # model.load_state_dict(torch.load('Models/Agent_'+str(a_num)+'/policy_model_Node#_' + str(node_num) + '_EP_' + str(num_episodes) + '.pth')) #+ str(node_num)
        # model.eval()


        # print("########################################################################")
        # print("########################################################################")
        # print("########################################################################")
        # print("########################################################################")
        # print("########################################################################")
        # print(f'For Agent {a_num} simulate a Markov Chain using the optimal policy:')
        
        # # Convert dictionary values to a list of arrays
        # # visited_states = [np.array(initial_states[0])]
        # visited_states = np.stack(visited_states)
    
        # #Initialize the environment.
        # env = Env(initial_states, # initial_state
        #          1, # total_episodes
        #          50, # show_path_num
        #          visited_states,  # visited_states
        #          available_actions, # basis_moves
        #          node_num, # node_num
        #          0, # P
        #          lb, #lb
        #          )
    
        # # reset environment and episode reward
        # state = env.reset()
     
        # # for each episode, only run 9999 steps so that we don't
        # # infinite loop while learning
        # for t in range(20):
    
        #     action_coeffs = select_best_action_transformer(model, state, action_space_values, t)
        #     # action = select_action_transformer(model, state, SavedAction, action_space_values, mask_range, t, None, True)
        #     # actions, action_probabilities = construct_stochastic_policy(model, state, SavedAction, action_space_values, mask_range, t, False, 0.01)
        #     # action_choices = []
        #     # for a in action_coeffs:
        #     action_rounded = np.array(np.round(action_coeffs), dtype=int)
        #     all_actions = [np.multiply(action_rounded[i], available_actions[i]) for i in range(len(action_rounded))]
        #     all_actions = np.stack(all_actions)
        #     action = np.sum(all_actions, 0)
            
                
        #     # action_choices, filtered_actions_probs = filter_actions(state, action_choices, action_probabilities)
        #     # if len(action_choices) > 1:
        #     #     random_action = np.random.randint(0,len(action_choices))
        #     # else:
        #     #     random_action = 0
        #     # print(random_action)
        #     # action = action_choices[random_action]
        #     # print("Action: ",action, " with probability ", filtered_actions_probs[random_action])
        #     print("Action: ", action)
        #     next_state = np.add(state, action)
        #     print("Next state: ", next_state)
        #     if all(coord >= 0 for coord in next_state):
        #         if next_state.tolist() not in visited_states.tolist():
        #             visited_states = np.concatenate((visited_states,[next_state]),axis=0)
        #         state = next_state
        #     print("#####################")
        
        # print(f'We discovered {visited_states.shape[0]} unique states using the optimal stochastic policy')



'''
load up all models and then for each state, with uniform probabilitiy, pick a an action \pi_{i}(S).
This will be our stochastic policy. 

'''