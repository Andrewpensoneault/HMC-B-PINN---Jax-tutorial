# Tutorial for posterior inference in Bayesian Physics Informed Neural Networks with HMC in Jax

In this tutorial, we will demonstrate how to perform posterior inference using Hamiltonian Monte Carlo (HMC) in Bayesian Physics Informed Neural Networks (PINNs). By using the Jax library, we will take advantage of its automatic differentiation and GPU acceleration capabilities to efficiently perform inference in Bayesian PINNs.

# Bayesian Physics-Informed Neural Networks (B-PINN) 

Bayesian Physics-Informed Neural Networks (B-PINN) is a machine learning method that combines Bayesian neural networks with physics-informed neural networks to solve inverse problems in scientific and engineering applications. B-PINN provides a powerful and flexible framework for integrating physical laws and experimental data into the training of neural networks, enabling accurate predictions and uncertainty quantification.

## How does B-PINN work?

We consider the following partial differential equation (PDE): 
$$
\begin{align}
\mathcal{N}_x(u(\mathbf{x});\boldsymbol{\lambda})&=f(\mathbf{x}) \quad \mathbf{x}\in \Omega,\\
\mathcal{B}_x(u(\mathbf{x});\boldsymbol{\lambda})&=b(\mathbf{x}) \quad \mathbf{x}\in \partial\Omega,
\end{align}
$$
where $\mathcal{N}_x$ and $\mathcal{B}_x$ denote the differential and boundary operators, respectively. The spatial domain $\Omega\subseteq\mathbb{R}^d$ has boundary $\Gamma$, and $\boldsymbol{\lambda}\in\mathbb{R}^{N_\lambda}$ represents a vector of unknown physical parameters. The forcing function $f(\mathbf{x})$ and boundary function $b(\mathbf{x})$ are assumed known, and $u(\mathbf{x})$ is the solution of the PDE. We approximate the solution $u(\mathbf{x})$ with a neural network approximation $\tilde{u}(\mathbf{x};\boldsymbol{\theta}).$ Here $\boldsymbol{\theta}$ represents the weights and biases of the neural network approximations. Additionally, we denote the quantity $\boldsymbol{\xi}=[\boldsymbol{\theta},\boldsymbol{\lambda}]$ to be the total set of physical and neural network parameters. In this setting, we have access to to $N_u$ measurements $$\mathcal{D}_u=\{(\mathbf{x}_u^i,u(\mathbf{x}_u^i))\}_{i=1}^{N_u}=\{(\mathbf{x}_u^i, u^i)\}_{i=1}^{N_u}$$ of the forward solution $u(\mathbf{x})$. Additionally, we utilize information from the PDE and boundary, denoted "residual points" and "boundary points," respectively, as follows:
$$
\begin{align}
    \mathcal{D}_f =\{(\mathbf{x}_f^i,f(\mathbf{x}_f^i))\}_{i=1}^{N_f}= \{(\mathbf{x}_f^i,f^i)\}_{i=1}^{N_f}\\
    \mathcal{D}_b =\{(\mathbf{x}_b^i,b(\mathbf{x}_b^i))\}_{i=1}^{N_b} = \{(\mathbf{x}_b^i,b^i)\}_{i=1}^{N_b},
\end{align} 
$$
with residual locations $\mathbf{x}_f^i\in \Omega$ and boundary locations $\mathbf{x}_b^i\in \partial\Omega$. Bayesian PINNs place the following assumptions on the likelihood functions $p(\mathcal{D}_u,\mathcal{D}_f,\mathcal{D}_b|\boldsymbol{\xi})$ of the three groups of measurements
$$
\begin{align}
p(\mathcal{D}_u,\mathcal{D}_f,\mathcal{D}_b|\boldsymbol{\xi})&=p(\mathcal{D}_u|\boldsymbol{\xi})p(\mathcal{D}_f|\boldsymbol{\xi})p(\mathcal{D}_b|\boldsymbol{\xi}),\\
p(\mathcal{D}_u|\boldsymbol{\xi}) &= \prod_{i=1}^{N_u}p(u^i|\boldsymbol{\xi}),\quad
p(\mathcal{D}_f|\boldsymbol{\xi}) = \prod_{i=1}^{N_f}p(f^i|\boldsymbol{\xi}),\quad
p(\mathcal{D}_b|\boldsymbol{\xi}) = \prod_{i=1}^{N_b}p(b^i|\boldsymbol{\xi}),\\
p(u^i|\boldsymbol{\xi})&=\frac{1}{\sqrt{2\pi\sigma_{\eta_u}^2}}\exp\left(-\frac{\left(u^i-\tilde{u}(\mathbf{x}_u^i;\boldsymbol{\theta})\right)^2}{2\sigma_{\eta_u}^2}\right),\\
p(f^i|\boldsymbol{\xi})&=\frac{1}{\sqrt{2\pi\sigma_{\eta_f}^2}}\exp\left(-\frac{\left(f^i-\mathcal{N}_x(\tilde{u}(\mathbf{x}_f^i;\boldsymbol{\theta});\boldsymbol{\lambda})\right)^2}{2\sigma_{\eta_f}^2}\right),\\
p(b^i|\boldsymbol{\xi})&=\frac{1}{\sqrt{2\pi\sigma_{\eta_b}^2}}\exp\left(-\frac{\left(b^i-\mathcal{B}_x(\tilde{u}(\mathbf{x}_b^i;\boldsymbol{\theta});\boldsymbol{\lambda})\right)^2}{2\sigma_{\eta_b}^2}\right).
\end{align}
$$
Here, $\sigma_{\eta_u}$ is the standard deviations of the forward measurements, which is assumed known a priori. Additionally, $\sigma_{\eta_f}$ and $\sigma_{\eta_b}$ are the standard deviations of the residual points and boundary points. While in this setting $f$ and $b$ are assumed known allowing us to draw noise-free samples, this choice of likelihood allows us to place soft constraints on the physics.

Additionally, the following form of the prior $p(\boldsymbol{\xi})$ is typically assumed
$$
\begin{align}
p(\boldsymbol{\xi})&= p(\boldsymbol{\lambda})p(\boldsymbol{\theta})\\
p(\boldsymbol{\theta}) &= \prod_{i=1}^{N_\theta} p(\theta^i), \quad
p(\theta^i) \sim \mathcal{N}\left(0,\sigma^i_\theta\right),\\
p(\boldsymbol{\lambda}) &= \prod_{i=1}^{N_\lambda} p(\lambda^i), \quad p(\lambda^i) \sim \mathcal{N}\left(0,\sigma^i_\lambda\right).
\end{align}
$$
Here  $\sigma_{\lambda}$, $\sigma_{\theta}$ are the prior physical and neural network parameter standard deviations, respectively. From here, we can formulate the posterior distribution $p(\boldsymbol{\xi}|\mathcal{D}_u,\mathcal{D}_f,\mathcal{D}_b)$ using Bayes theorem
$$
\begin{align}
p(\boldsymbol{\xi}|\mathcal{D}_u,\mathcal{D}_f,\mathcal{D}_b)&\propto p(\boldsymbol{\xi})p(\mathcal{D}_u,\mathcal{D}_f,\mathcal{D}_b|\boldsymbol{\xi}),
\end{align}
$$
and we may apply any inference method if interest to perform inference on the physical and nerual network parameters.

![B-PINN Flow Chart](https://drive.google.com/uc?id=1UvxadtquFBS_F7wIOY6NRJyTHa0poXRp)

# B-PINN Advantages
Compared to other inverse problem solvers, B-PINN has several advantages:

- It can handle noisy and incomplete experimental data by providing probabilistic estimates of the solution and its uncertainty.
- It can incorporate physical laws and constraints into the training of neural networks, enabling accurate predictions and better generalization.
- It can handle complex and nonlinear systems, as neural networks are universal approximators.
- It can reduce the computational cost of solving inverse problems, as the physics-informed neural network can provide accurate and fast surrogate models.

# Limitations of B-PINN
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
