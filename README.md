# ME3302 Warehouse Management

This repository contains implementations of various multi-agent reinforcement learning approaches for warehouse management and task allocation.

## Project Overview

We've developed a custom warehouse environment with multiple agents that need to coordinate to pick up and deliver packages efficiently. The project explores different approaches to solving this complex coordination problem.

## Key Components

### Environment
- `warehouse_env.py`: A custom Gymnasium environment that simulates a warehouse with shelves, packages, and multiple agents. Supports different observation types (centralized, decentralized, partially observable) and provides a flexible reward structure for training different algorithms.

### Implemented Approaches

#### Classical Methods
- `A_Star.py`: Implementation of A* pathfinding algorithm with proximity-based task allocation. This serves as a baseline approach that doesn't use learning but provides efficient paths between locations.

#### Reinforcement Learning Methods
- `PPO_training.py`: Centralized Proximal Policy Optimization implementation for warehouse management. Treats all agents as a single entity with a shared policy.

- `multi_agent_decentralized.py`: Implements a decentralized multi-agent approach where each agent makes decisions based on its own observations while sharing a critic that has access to the full state.

- `maddpg.py`: Multi-Agent Deep Deterministic Policy Gradient implementation, which allows agents to learn cooperative behaviors through a centralized training, decentralized execution paradigm.

- `HRL_DQ.py`: Hierarchical Reinforcement Learning with Deep Q-Networks, which decomposes the problem into high-level options (like "go to pickup location") and low-level actions (like "move forward").

#### Evaluation
- `metrics.py`: Utilities for calculating performance metrics such as throughput, agent utilization, task completion time, and collision rates.

## Comparative Analysis

Our work explores the trade-offs between different approaches:
- Classical methods provide reliable performance but lack adaptability
- Centralized methods simplify learning but don't scale well with more agents
- Decentralized methods scale better but face coordination challenges
- Hierarchical methods provide a good balance of coordination and scalability

Each approach has been evaluated on metrics including delivery throughput, agent utilization, task completion time, and collision avoidance.

## Future Work

- Improving scalability to larger warehouse environments
- Incorporating more realistic constraints like battery management
- Exploring transfer learning between different warehouse layouts
- Implementing more sophisticated communication mechanisms between agents