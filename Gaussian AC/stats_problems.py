import numpy as np
import re
from helper_functions import extract_lattice_basis_sparse, convert_sym_to_np




def mc2_matrix_to_np_arr(matrix_text_file):

    A_mc2 = open(matrix_text_file, "r") # open the file.
    dimensions = A_mc2.readline() # read the first line.
    dim = re.findall(r'\d+', dimensions)
    n = int(dim[0])
    m = int(dim[1])

    A = np.zeros((n,m),dtype=int) # define the matrix in numpy.

    for i in range(n):
        row = A_mc2.readline()
        row_detected = re.findall('[-+]?\d+', row)
        for j in range(m):
            A[i,j] = int(row_detected[j])
    return A




# Sparse table with large entries.
def return_stats_problem_1():
    
    node_num = 4
    
    initial_states = {}
    initial_states[0] = np.array([10, 0, 0, 2, 0, 3, 0 ,40, 10, 0, 2, 0, 0, 3, 40, 0])
    
    
    available_actions = [ [1,-1,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,],
                    [0,1,-1,0, 0,-1,1,0, 0,0,0,0, 0,0,0,0], 
                    [0,0,1,-1,0,0,-1,1,0,0,0,0,0,0,0,0],  
                    [0,0,0,0,1,-1,0,0,-1,1,0,0,0,0,0,0],
                    [0,0,0,0,0,1,-1,0, 0,-1,1,0,0,0,0,0], 
                    [0,0,0,0,0,0,1,-1, 0,0,-1,1,0,0,0,0], 
                    [0,0,0,0,0,0,0,0, 1,-1,0,0,-1,1,0,0],
                    [0,0,0,0,0,0,0,0,0,1,-1,0,0,-1,1,0],
                    [0,0,0,0,0,0,0,0,0,0,1,-1,0,0,-1,1] ]
    
    
    design_mat = np.array([[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                       [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                       [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]])
    
    margin = np.dot(design_mat, initial_states[0])
    
#     available_actions = extract_lattice_basis_sparse(design_mat) # get the lattice basis out of the design matrix.
#     available_actions = convert_sym_to_np(available_actions) # convert to numpy.
    
    return node_num, initial_states, available_actions, design_mat, margin





# Sparse table with large entries.
def return_stats_problem_2():
    
    node_num = 4
    
    initial_states = {}
    initial_states[0] = np.array([1, 1, 0, 0, 0, 1, 1 , 0, 0, 0, 1, 1, 1, 0, 0, 1])
    
    
#     available_actions = [ [1,-1,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,],
#                     [0,1,-1,0, 0,-1,1,0, 0,0,0,0, 0,0,0,0], 
#                     [0,0,1,-1,0,0,-1,1,0,0,0,0,0,0,0,0],  
#                     [0,0,0,0,1,-1,0,0,-1,1,0,0,0,0,0,0],
#                     [0,0,0,0,0,1,-1,0, 0,-1,1,0,0,0,0,0], 
#                     [0,0,0,0,0,0,1,-1, 0,0,-1,1,0,0,0,0], 
#                     [0,0,0,0,0,0,0,0, 1,-1,0,0,-1,1,0,0],
#                     [0,0,0,0,0,0,0,0,0,1,-1,0,0,-1,1,0],
#                     [0,0,0,0,0,0,0,0,0,0,1,-1,0,0,-1,1] ]
    
    
    design_mat = np.array([[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                       [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                       [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
                       [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
                       [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]])
    
    margin = np.dot(design_mat, initial_states[0])
    
    available_actions = extract_lattice_basis_sparse(design_mat) # get the lattice basis out of the design matrix.
    available_actions = convert_sym_to_np(available_actions) # convert to numpy.
    
    return node_num, initial_states, available_actions, design_mat, margin





# Sparse table with large entries.
def return_stats_problem_3():
    
    node_num = 4
    
    initial_states = {}
    
    initial_states[0] = np.array([42, 55, 22, 3, 72, 82, 60, 12, 90, 106, 85, 25, 27, 48, 47, 8, 8, 18, 19, 5, 1, 2, 8, 19, 1, 2, 15, 33, 2, 5, 25, 83, 2, 2, 10, 45, 0, 0, 12, 19, 0, 0, 1, 19, 0, 0, 3, 60, 0, 0, 5, 86, 0, 0, 2, 36, 0, 0, 1, 14, 172, 151, 107, 42, 208, 198, 206, 92, 279, 271, 331, 191, 99, 126, 179, 97, 36, 35, 99, 79])

    
    
    path = 'C:/Users/gvozd/Desktop/University/Research/Code/Combinatorial RL/Fiber Sampling using RL/Deep Fiber Sampling/Actor Critic/Gaussian AC/Real Data/StatsProblem/DobraDesignMat.txt'
    design_mat = mc2_matrix_to_np_arr(path)
    

    
    margin = np.dot(design_mat, initial_states[0])
    
    available_actions = extract_lattice_basis_sparse(design_mat) # get the lattice basis out of the design matrix.
    available_actions = convert_sym_to_np(available_actions) # convert to numpy.
    
    return node_num, initial_states, available_actions, design_mat, margin







# initial_states[0] = np.array([5, 0, 2, 1, 5, 1, 0, 0, 4, 1, 0, 0, 6, 0, 2, 0, 8, 0, 11, 0, 13, 0, 1, 0, 3, 0, 1, 0, 26, 0, 1, 0, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 8, 2, 6, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 17, 10, 1, 1, 16, 7, 0, 0, 0, 2, 0, 0, 10, 6, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 7, 3, 1, 1, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 3, 2, 0, 23, 4, 0, 0, 22, 2, 0, 0, 57, 3, 0, 0, 5, 1, 0, 0, 11, 0, 1, 0, 11, 0, 0, 0, 29, 2, 1, 1, 3, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 25, 0, 1, 37, 26, 0, 0, 15, 10, 0, 0, 43, 22, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 4, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




