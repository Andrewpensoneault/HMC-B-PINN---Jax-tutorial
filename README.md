# Tutorial for posterior inference in Bayesian Physics Informed Neural Networks with HMC in Jax

In this tutorial, we will demonstrate how to perform posterior inference using Hamiltonian Monte Carlo (HMC) in Bayesian Physics Informed Neural Networks (PINNs). By using the Jax library, we will take advantage of its automatic differentiation and GPU acceleration capabilities to efficiently perform inference in Bayesian PINNs.

# Bayesian Physics-Informed Neural Networks (B-PINN) 

Bayesian Physics-Informed Neural Networks (B-PINN) is a machine learning method that combines Bayesian neural networks with physics-informed neural networks to solve inverse problems in scientific and engineering applications. B-PINN provides a powerful and flexible framework for integrating physical laws and experimental data into the training of neural networks, enabling accurate predictions and uncertainty quantification.

## How does B-PINN work?
![B-PINN Flow Chart](https://drive.google.com/uc?id=1UvxadtquFBS_F7wIOY6NRJyTHa0poXRp)

## B-PINN Advantages
Compared to other inverse problem solvers, B-PINN has several advantages:

- It can handle noisy and incomplete experimental data by providing probabilistic estimates of the solution and its uncertainty.
- It can incorporate physical laws and constraints into the training of neural networks, enabling accurate predictions and better generalization.
- It can handle complex and nonlinear systems, as neural networks are universal approximators.
- It can reduce the computational cost of solving inverse problems, as the physics-informed neural network can provide accurate and fast surrogate models.

## Limitations of B-PINN
Despite its many advantages, B-PINN also has some limitations that should be taken into consideration:

- The training process of B-PINN can be computationally expensive, particularly when dealing with large datasets or high-dimensional parameter spaces. This may require the use of specialized hardware or distributed computing resources.
- The choice of the prior distributions for the physical and neural network parameters can have a significant impact on the performance of B-PINN. 
- The choice of hyperparameters for the likelihood function can be challenging, particularly for the boundary and residual points when the physics information is assumed to be noise-free.

# Hamiltonian Monte Carlo
Hamiltonian Monte Carlo (HMC) is a powerful Markov chain Monte Carlo (MCMC) algorithm for sampling from distributions that is commonly used for inference with Bayesian Neural Networks (BNNs). HMC is also known as Hybrid Monte Carlo because it combines the Metropolis-Hastings algorithm with Hamiltonian dynamic.

## How HMC works
HMC samples from the target distribution by proposing new states of a chain using a Hamiltonian dynamics simulation. The Hamiltonian dynamics are defined by a Hamiltonian function that describes the energy of the system being sampled.

At each iteration, HMC chooses a random starting point in the state space and then simulates the dynamics of the system by solving the Hamiltonian equations of motion using a numerical integration scheme such as leapfrog integration. The simulated dynamics are used to propose a new state of the chain. The proposal is accepted or rejected using the Metropolis-Hastings acceptance probability that depends on the ratio of the probabilities of the proposed state and the current state.

## Advantages of HMC
Compared to other MCMC algorithms such as the random walk Metropolis algorithm, HMC has several advantages:

- It generates proposals that are highly correlated with the current state, resulting in fewer rejections and faster convergence to the target distribution.
- It can explore the target distribution more efficiently by exploiting the geometry of the distribution.
- It can sample from distributions with complex geometries that are difficult for other MCMC algorithms.

## Limitations of HMC
Although HMC has several advantages, it has some limitations:

- It requires specifying the Hamiltonian function, which can be difficult for some distributions.
- The choice of the integration step size and the number of integration steps can significantly affect the performance of the algorithm. These parameters need to be carefully tuned to ensure good performance.
- The computational cost of simulating the dynamics can be high, especially for high-dimensional distributions.
