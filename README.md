# Hamiltonian Monte Carlo -- Tutorial
Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) algorithm that is particularly effective in sampling from distributions that have complex geometries such as those with multiple modes, curved or elongated tails. HMC is also known as Hybrid Monte Carlo as it combines the Metropolis-Hastings algorithm with Hamiltonian dynamics from physics.

## How HMC works
HMC samples from the target distribution by proposing new states of a chain using a Hamiltonian dynamics simulation. The Hamiltonian dynamics are defined by a Hamiltonian function that describes the energy of the system being sampled.

At each iteration, HMC chooses a random starting point in the state space and then simulates the dynamics of the system by solving the Hamiltonian equations of motion using a numerical integration scheme such as leapfrog integration. The simulated dynamics is used to propose a new state of the chain. The proposal is accepted or rejected using the Metropolis-Hastings acceptance probability that depends on the ratio of the probabilities of the proposed state and the current state.

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
