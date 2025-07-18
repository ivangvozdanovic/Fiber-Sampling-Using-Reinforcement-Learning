# Fiber-Sampling-Using-Reinforcement-Learning

The code in this repo is for the paper: "Learning to sample fibers for goodness-of-fit testing". (https://arxiv.org/abs/2405.13950)

We consider the problem of constructing exact goodness-of-fit tests for discrete exponential family models. This classical problem remains practically unsolved for many types of structured or sparse data, as it rests on a computationally difficult core task: to produce a reliable sample from lattice points in a high-dimensional polytope. We translate the problem into a Markov decision process and demonstrate a reinforcement learning approach for learning `good moves' for sampling. We illustrate the approach on data sets and models for which traditional MCMC samplers converge too slowly due to problem size, sparsity structure, and the requirement to use prohibitive non-linear algebra computations in the process. The differentiating factor is the use of scalable tools from \emph{linear} algebra in the context of theoretical guarantees provided by \emph{non-linear} algebra. Our algorithm is based on an actor-critic sampling scheme, with provable convergence.
The discovered moves can be used to efficiently obtain an exchangeable sample, significantly cutting computational times with regards to statistical testing.


One of the examples we consider is the coauthorship dataset. This dataset is given in the form a graph where each node is an author and there exists an edge between two nodes if the two corresponding authors worked on a paper together. We are interested in verifying whether the data follows the beta model.


<p float="left">
  <img src="images/Rplot-Coauthor-NextComponent.png" width="45%" />
  <img src="images/ChiSquare_Node_23.png" width="45%" />
</p>


How to use the code:

- Download the whole repository and run the Gaussian A2C Fiber Sampling notebook.
- In this notebook you can define sampling for 4 different problems. For custome problems, you need to define the design matrix $A$, initial solution $x_0$ and margin $Ax = b$. Then extract the lattice basis.
- After computing the lattice basis, simply run the trainig cell and the RL will start training.
- In the end of the notebook, you can load the trained policy, rerun it on the same fiber and compute the random sample from the fiber.
- Finally, the code computes the emprical chi-square distribution for the given sample.
