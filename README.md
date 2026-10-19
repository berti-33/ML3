<div align="center">

  <h1>🧬 JAX-2048: Hardware-Accelerated Neuroevolution</h1>
  <p><strong>A fully vectorized, differentiable 2048 game engine and evolutionary neural agent running end-to-end on GPU/TPU with JAX.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/Language-Python%203.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Framework-JAX-blue?style=for-the-badge&logo=google&logoColor=white" alt="JAX" />
    <img src="https://img.shields.io/badge/Algorithm-Genetic%20Algorithm%20%2F%20ES-orange?style=for-the-badge" alt="Algorithm" />
    <img src="https://img.shields.io/badge/Device-GPU%20%2F%20TPU%20Ready-green?style=for-the-badge" alt="Hardware" />
  </p>

  <p>
    <a href="#-key-features">Key Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-how-it-works">How It Works</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-hyperparameters">Hyperparameters</a> •
    <a href="#-known-optimizations--future-work">Roadmap</a>
  </p>

</div>

---

## ⚡ Key Features

- **Pure Functional Vectorization (`jax.vmap`):** Simulates hundreds of independent 2048 board states and neural forward passes in parallel with zero native Python loop bottlenecks.
- **Traceable Game Logic:** Complete game rules (board rotation, row compression, sliding tile merges, random tile spawns, and termination checks) implemented using functional control flow (`jax.lax.cond`, `jax.lax.switch`, `jax.lax.fori_loop`, and `jax.lax.while_loop`).
- **End-to-End On-Device Execution:** From board initialization to selection, crossover, and Gaussian mutation, all states remain in VRAM as persistent JAX PyTrees.
- **Tournament Selection & Elitism:** Preserves high-performing parameter sets across generations and scores agents using multi-game stochastic evaluation to reduce random spawn variance.

---

### 1. Vectorized Game Engine (`move`, `leftM`, `merge`)
The core sliding and merging logic compiles down into static computational graphs:
- **Left Slide (`leftM`):** Vectorized row-by-row via `jax.vmap(merge)`.
- **Directional Transformations:** Up, Right, and Down slides are derived by applying functional matrix transformations (`board.T`, `jnp.flip`) followed by `leftM`.
- **Fall-back Moves:** The network outputs a priority order for all 4 possible actions (`[up, right, down, left]`). The engine executes the highest-ranked action that causes a state mutation (`board != board0`).
- 
## 🧠 Evolutionary Training Loop

The optimization uses a **Genetic Algorithm (GA)** to evolve neural network weights without computing analytical gradients:

1. **Fitness Evaluation:** Each individual policy plays $K=4$ independent games initialized with distinct PRNG seeds. Fitness is evaluated as the mean game score:
   $$\text{Fitness}_i = \frac{1}{4} \sum_{k=1}^{4} \text{Score}_{i, k}$$
2. **Elitism:** The individual that achieves the all-time maximum score (`xbest`) is tracked and injected directly into future generations.
3. **Tournament Selection:** Pairs of networks are sampled from the population; the network with higher average fitness survives.
4. **Gaussian Mutation (`a3`):** Synaptic weights are perturbed by scaled Gaussian noise:
   $$W_{l}' = W_l + \mathcal{N}(0, 1) \times \frac{\max(W_l) - \min(W_l) + \epsilon}{100} \odot \mathbb{I}_{(\text{rand} < p_{\text{mut}})}$$

---
