def reward_cost(reward_list, state):  # reward for fiding a table which minimizes cost.
    reward = 0
    for i in range(len(state)):
        reward -= state[i]*reward_list[i]
    return reward




def calculate_reward1(objective, state):
        L1_sum = 0
        reward = 0
        for i in objective:
            L1_sum += abs(state[i])
        if L1_sum == 0:
            reward = 100
        else:
            reward = 1/L1_sum  # The smaller the distance between the objective table and current table, L1 sum is smaller 
                               # so 1/L1 is getting bigger. Reward the agent when he is close to the objective table.
        return reward
        
def calculate_reward2(objective,state):
    reward = 0
    for i in range(len(objective)):
        reward += objective[i]*state[i]

    return reward
    
def calculate_reward3(objective, state):
    reward = 0
    for i in range(len(objective)):
        reward += objective[i]*state[i]

    diam = 0
    state_len = len(state)
    for n in range(agent_num):
        for m in range(agent_num):
            diff = np.max(np.subtract(state[n],state[m]))
            diam += diff

    reward += 1/diam