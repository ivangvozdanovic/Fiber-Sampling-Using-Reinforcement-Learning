from DeepFiberSamplingGaussianENV import PolytopeENV as Env 
from ipfn import ipfn
from GaussianA2C import select_action_transformer
import numpy as np
import matplotlib.pyplot as plt




def simulate_trajectory(model,
                        SavedAction,
                        env,
                        action_space_values,
                        max_path_length,
                        mask_range):
                                
    path_reward = []
    step = 0
    path = []
    

    state = env.reset()
    done = False
    
    
    for i in range(max_path_length):
        
        action = select_action_transformer(model, state, SavedAction, action_space_values, mask_range, i, True)
        next_state, reward, done, info = env.step(action)
       
#         print("Current state: ", state, "--|||-- Current state cost: ", reward, " and action coeffs: ", action)
        state = next_state
        
    return state



def find_exchangable_sample(model,
                            SavedAction,
                            initial_state,
                            episode_num,
                            max_path_length,
                            visited_states,
                            available_actions,
                            action_space_values,
                            mask_range,
                            node_num,
                            lb,
                            ub,
                            chain_num):
    
    
    computed_sample = []
    
    #Initialize the environment.
    env = Env(initial_state, # initial_state
             episode_num, # total_episodes
             max_path_length,
             50, # show_path_num
             visited_states,  # visited_states
             available_actions, # basis_moves
             node_num, # node_num
             0, # P
             lb, #lb
             False)
    simulation_root_state = simulate_trajectory(model,
                                                SavedAction,
                                                env,
                                                action_space_values,
                                                max_path_length,
                                                mask_range)
    for sim in range(chain_num):
        
        #Initialize the environment.
        env = Env(initial_state, # initial_state
                 episode_num, # total_episodes
                 max_path_length,
                 50, # show_path_num
                 visited_states,  # visited_states
                 available_actions, # basis_moves
                 node_num, # node_num
                 0, # P
                 lb, #lb
                 False)
        
        sample_state = simulate_trajectory(model,
                                           SavedAction,
                                           env,
                                           action_space_values,
                                           max_path_length,
                                           mask_range)
        computed_sample.append(sample_state)
        
    return computed_sample





# Compute the chi square statistic for the exchangable sample and plots the empirical distribution.
def compute_chi_square(problem_name, init_sol, margin, sample, nodes, num_episodes):

    keys = []
    values = []
    frequencey_dict = {}
    extrem_obs_count = 0

    #Compute the expected table in the fiber.
    m = np.ones((len(nodes),len(nodes)))
    xip = np.array(margin)
    xpj = np.array(margin)
    aggregates = [xip, xpj]
    dimensions = [[0], [1]]
    IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-8)
    m = IPF.iteration()
    m_round = np.round(m,2)




    # Compute the expected table in the fiber
#     m = np.ones((4, 5, 4))  

#     margin = np.array(margin)  # Convert to NumPy array if not already
# #     margin = np.where(margin == 0, 0.001, margin)  # Replace all 0s with 0.001
#     print("Margin: ", margin)
#     # Split the margin vector into the three parts
    
#     # First part (20 values), corresponds to x1
#     x1_marginal = margin[:20].reshape(4, 5)  # This corresponds to the first 20 values
#     # Second part (16 values), corresponds to x2
#     x2_marginal = margin[20:36].reshape(4, 4)  # The next 16 values
#     # Third part (20 values), corresponds to x3
#     x3_marginal = margin[36:].reshape(5, 4)  # The final 20 values

#     # Prepare the aggregates and dimensions for IPF
#     aggregates = [x1_marginal, x2_marginal, x3_marginal]
#     dimensions = [[0, 1], [0, 2], [1, 2]]  # Same as before

#     # Perform Iterative Proportional Fitting (IPF)
#     IPF = ipfn.ipfn(m, aggregates, dimensions, convergence_rate=1e-8)

#     # Perform iterations and round the result
#     m = IPF.iteration()
#     m_round = np.round(m, 2)

    # Print the result
    print(m_round.shape, m_round)
    exp_table = m_round.flatten(order='C')
    print("Expected table: \n", exp_table,exp_table.shape)
    structural_zero_indices = np.where(exp_table == 0)[0]
    
    # Compute chi-square for u_0.
    init_sol = np.delete(init_sol, structural_zero_indices) # remove the structural zeros
    exp_table = np.delete(exp_table, structural_zero_indices) # remove the structural zeros
#     initial_chi_square = np.round( np.sum( np.divide(np.power( np.subtract(np.array(init_sol), exp_table), 2), exp_table) ), 2)
    initial_chi_square = np.round( np.sum( np.divide(np.power( np.subtract(np.array(init_sol), exp_table), 2), exp_table) ), 2)
    print("Initial Chi-square: ", initial_chi_square)

    for i in range(len(sample)):
        values_at_indices = sample[i][structural_zero_indices]
        if np.all(values_at_indices != 0):
            continue
        else:
            sample[i] = np.delete(sample[i], structural_zero_indices)
            chi_s = np.round( np.sum( np.divide(np.power( np.subtract(np.array(sample[i]), exp_table), 2), exp_table) ), 2)
            chi_s = np.round(chi_s, 2)

            if chi_s >= initial_chi_square:
                extrem_obs_count += 1

            key = chi_s
            print(chi_s)
            if key in keys:
                frequencey_dict[key] += 1
            else:
                keys.append(key)
                frequencey_dict[key] = 1

    # Compute the P value.
    print("The p-value of seeing obs as least as extreme as u_0: ", extrem_obs_count/len(sample))

    # Prepare histogram colors and widths.
    # Plot the emperical chi-square distribution.
    
    frequencey_dict = dict(sorted(frequencey_dict.items()))


    colors = []
    widths = []

#     for k in list(frequencey_dict.keys()):
#         if k == initial_chi_square:
#             frequencey_dict[initial_chi_square] = 3000
#             widths.append(1.3)
#             colors.append('red')
#         else:
#             widths.append(1)
#             colors.append('black')
            
#     key_list = list(frequencey_dict.keys())
#     key_length = len(key_list)

    # Define the interval for displaying x-ticks (for example, show every 5th tick)
#     interval = max(1, key_length // 10)
#     tick_num = key_length//5
#     custom_x_values = [keys[i] for i in range(0,len(key_list),tick_num)]#list(range(min_key, max_key, interval))
#     custom_x_values = sorted(custom_x_values)
#     l = [frequencey_dict[i] for i in custom_x_values]
#     custom_x_labels = [str(i) for i in custom_x_values]
#     plt.bar(frequencey_dict.keys(),frequencey_dict.values(), color=colors, width = widths)
#     plt.xticks(ticks=custom_x_values, labels=custom_x_labels, rotation=45, ha='right', size='small')
#     plt.xlabel('Chi-Square Observations')
#     plt.ylabel('Frequency')
#     plt.title('Empirical Distribution')
#     plt.grid(visible=True)

#     plt.savefig("Figures/ChiSqure_"+problem_name+"_Node#_"+str(len(nodes))+"_EP_"+ str(num_episodes)+".png"")
#     plt.show()
    
    return extrem_obs_count/len(sample)