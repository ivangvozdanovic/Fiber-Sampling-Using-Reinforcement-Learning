import random
import numpy as np
from multiprocessing import Process, Queue

from DeepFiberSamplingENV import PolytopeENV 

def log_writer(log_queue):
    """ Continuously read log messages from the queue and print them """
    while True:
        message = log_queue.get()
        if message == "DONE":
            break
        print(message)

def run_agent(env_params, log_queue):
    env = PolytopeENV(*env_params, log_queue=log_queue)
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
    log_queue.put("DONE")

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    # Define parameters
    initial_states = {0: np.array([0, 1, 2])}  # Example initial states; modify as needed
    visited_states = np.array([[0, 1, 2]])
    basis_moves = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    env_params = (
        initial_states,  # initial states
        10,  # total episodes
        50,  # show path num
        visited_states,  # visited states
        basis_moves,  # basis moves
        5,  # node number
        0,  # P
        0,  # lb
    )

    # Set up logging
    log_queue = Queue()
    log_process = Process(target=log_writer, args=(log_queue,))
    log_process.start()

    # Run agent
    agent_process = Process(target=run_agent, args=(env_params, log_queue))
    agent_process.start()
    agent_process.join()

    # Signal the log process to finish
    log_queue.put("DONE")
    log_process.join()