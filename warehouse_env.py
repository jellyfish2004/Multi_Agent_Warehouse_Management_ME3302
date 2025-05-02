import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import time

class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=(15, 9), num_agents=2, num_shelves=10, shelf_spacing=2, shelf_length=4, poisson_lambda=0.1, observation_type='centralized', normalize=True, max_requests=5, goal_deliveries=10):
        super().__init__()
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_shelves = num_shelves
        self.shelf_spacing = shelf_spacing
        self.shelf_length = shelf_length
        self.poisson_lambda = poisson_lambda
        self.observation_type = observation_type
        self.normalize = normalize
        # 0: empty, 1: shelf, 2: destination, 3: agent
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.agent_positions = []
        self.agent_orientations = []  # 0: up, 1: right, 2: down, 3: left
        self.agent_carrying = [None for _ in range(num_agents)]
        self.requests = []  # Will store (request_id, src, dest, picked)
        self.max_requests = max_requests
        self.goal_deliveries = goal_deliveries
        self.steps_taken = 0
        self.deliveries_completed = 0
        self.next_request_id = 0  # Counter for generating unique request IDs
        self.num_collisions = 0
        self.requests_static = []
        
        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([7] * num_agents)  # Updated to include idle action
        """
        0: Move forward  
        1: Move backward  
        2: Rotate left  
        3: Rotate right  
        4: Pick box  
        5: Place box
        6: Idle
        """
        
        # Calculate observation vector size based on observation type
        if observation_type == 'centralized':
            # Grid + agents info + requests info
            obs_size = (grid_size[0] * grid_size[1]) + (num_agents * 4) + (self.max_requests * 6)
            self.observation_space = spaces.Box(low=-1, high=max(grid_size[0], grid_size[1], 3, self.max_requests), 
                                            shape=(obs_size,), dtype=np.float64)
            self.structured_observation_space = spaces.Dict({
                'grid': spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.int8),
                'agents': spaces.Box(low=0, high=3, shape=(num_agents, 4), dtype=np.int8),
                'requests': spaces.Box(low=0, high=1, shape=(self.max_requests, 6), dtype=np.int8)
            })

        elif observation_type == 'decentralized':
            # Grid + agent info + requests info
            obs_size = (grid_size[0] * grid_size[1]) + 4 + (self.max_requests * 6) # per agent
            self.observation_space = spaces.Box(low=-1, high=max(grid_size[0], grid_size[1], 3, self.max_requests), 
                                            shape=(num_agents, obs_size), dtype=np.int8)
            self.structured_observation_space = spaces.Dict({
                'grid': spaces.Box(low=0, high=1, shape=(grid_size[0], grid_size[1]), dtype=np.int8),
                'agents': spaces.Box(low=0, high=3, shape=(num_agents, 4), dtype=np.int8),
                'requests': spaces.Box(low=0, high=1, shape=(self.max_requests, 6), dtype=np.int8)
            })
            
        elif observation_type == 'decentralized_partial_observability':
            # 5x5 partial grid + agent info + requests info
            obs_size = (25 + 4 + (self.max_requests * 6)) # per agent
            self.observation_space = spaces.Box(low=-1, high=max(grid_size[0], grid_size[1], 3, self.max_requests), 
                                            shape=(num_agents, obs_size), dtype=np.int8)
            self.structured_observation_space = spaces.Dict({
                'grid': spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
                'agents': spaces.Box(low=0, high=3, shape=(num_agents, 4), dtype=np.int8),
                'requests': spaces.Box(low=0, high=1, shape=(self.max_requests, 6), dtype=np.int8)
            })
            
        else:
            raise ValueError(f"Unknown observation type: {observation_type}")

        # Initialize figure and axes for rendering
        self.fig = None
        self.ax = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid.fill(0)
        self._place_shelves()
        self._place_destinations()
        self._spawn_agents()
        self.requests = []
        self.requests_static = []
        self.num_collisions = 0
        self.steps_taken = 0
        self.deliveries_completed = 0
        self.next_request_id = 0  # Reset request ID counter
        self._update_grid()
        obs_vector, obs_structured = self._get_obs()
        self.agent_carrying = [None for _ in range(self.num_agents)]
        return obs_vector, obs_structured

    def _place_shelves(self):
        col = 1
        for _ in range(self.num_shelves):
            l = 0
            for row in range(1, self.grid_size[0] - 2):
                l += 1
                if l >= self.shelf_length:
                    l = 0
                else:
                    self.grid[row, col] = 1
            col += self.shelf_spacing
            if col >= self.grid_size[1] - 1:
                break

    def _place_destinations(self):
        # divide the grid bottom edge into self.num_agents-1 equal parts and place num_agents destinations there
        self.destinations = [(self.grid_size[0] - 1, i) for i in range(0, self.grid_size[1], (self.grid_size[1]-1) // (self.num_agents - 1))]
        for x, y in self.destinations:
            self.grid[x, y] = 2

    def _update_grid(self):
        self.grid.fill(0)
        self._place_shelves()
        self._place_destinations()
        for i, (x, y) in enumerate(self.agent_positions):
            self.grid[x, y] = 3
        for req in self.requests:
            req_id, src, dest, picked = req
            if not picked:
                self.grid[src[0], src[1]] = 4


    def _spawn_agents(self):
        self.agent_positions = []
        self.agent_orientations = []
        for _ in range(self.num_agents):
            while True:
                x = random.randint(0, self.grid_size[0] - 1)
                y = random.randint(0, self.grid_size[1] - 1)
                if self.grid[x, y] == 0:
                    self.agent_positions.append((x, y))
                    self.agent_orientations.append(random.randint(0, 3))
                    self.grid[x, y] = 3
                    break

    def _poisson_requests(self):
        if len(self.requests) >= self.max_requests:
            return
        if self.deliveries_completed + len(self.requests) >= self.goal_deliveries:
            return
        if np.random.rand() < self.poisson_lambda:
            shelf_cells = list(zip(*np.where(self.grid == 1)))
            src = random.choice(shelf_cells)
            dest = random.choice(self.destinations)
            # Add request ID to the request tuple
            self.requests.append((self.next_request_id, src, dest, False))  # False = not yet picked
            for req in self.requests_static:
                if req[0] == -1:
                    req[0] = self.next_request_id
                    req[1] = src
                    req[2] = dest
                    req[3] = False
                    break
            self.next_request_id += 1

    def step(self, actions):
        self._poisson_requests()
        rewards = []
        prev_num_requests = len(self.requests)
        
        for i, action in enumerate(actions):
            rewards.append(self._apply_action(i, action))
        
        self._update_grid()

        for i, pos in enumerate(self.agent_positions):
            if self.agent_positions.count(pos) > 1:
                self.num_collisions += 1
                rewards[i] += -20
        
        # Track deliveries completed
        if prev_num_requests > len(self.requests):
            self.deliveries_completed += prev_num_requests - len(self.requests)
            # reward all agents if any delivery is completed
            for i in range(self.num_agents):
                rewards[i] += 4
        
        self.steps_taken += 1
        terminated = [False] * self.num_agents
        if self.deliveries_completed == self.goal_deliveries:
            terminated = [True] * self.num_agents
        truncated = True if self.steps_taken >= 1500 else False
        truncated = [truncated] * self.num_agents
        
        obs_vector, obs_structured = self._get_obs()

        if self.observation_type == 'centralized':
            rewards = np.sum(rewards) / self.num_agents
            truncated = True if self.steps_taken >= 1500 else False
            terminated = True if self.deliveries_completed == self.goal_deliveries else False

        return obs_vector, rewards, terminated, truncated, obs_structured

    def _apply_action(self, agent_id, action):
        x, y = self.agent_positions[agent_id]
        ori = self.agent_orientations[agent_id]

        dx_dy = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = dx_dy[ori]
        fx, fy = x + dx, y + dy

        # print(f"Agent {agent_id}: Facing: {ori}, Action: {action}, Next: {(fx, fy)}")
        reward = 0
        prev_distance = None
        new_pos = (x,y)
        
        # If agent is carrying a box, compute distance to drop-off
        if self.agent_carrying[agent_id] is not None:
            req_id = self.agent_carrying[agent_id]
            for r_id, _, dest, _ in self.requests:
                if r_id == req_id:
                    prev_distance = abs(x - dest[0]) + abs(y - dest[1])
                    break

        if action == 0:  # move forward
            if 0 <= fx < self.grid_size[0] and 0 <= fy < self.grid_size[1] and self.grid[fx, fy] == 0:
                self.agent_positions[agent_id] = (fx, fy)
                reward += -1
                new_pos = (fx, fy)
            elif not (0 <= fx < self.grid_size[0] and 0 <= fy < self.grid_size[1]):
                reward += -10
            elif self.grid[fx, fy] == 1 or self.grid[fx, fy] == 2 or self.grid[fx, fy] == 4:
                self.num_collisions += 1
                reward += -10
            else:
                reward += -1
            
        elif action == 1:  # move backward
            bx, by = x - dx, y - dy
            if 0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1] and self.grid[bx, by] == 0:
                self.agent_positions[agent_id] = (bx, by)
                reward += -1
                new_pos = (bx, by)
            elif not (0 <= bx < self.grid_size[0] and 0 <= by < self.grid_size[1]):
                reward += -10
            elif self.grid[bx, by] == 1 or self.grid[bx, by] == 2 or self.grid[bx, by] == 4:
                self.num_collisions += 1
                reward += -10
            else:
                reward += -1

        elif action == 2:  # rotate left
            self.agent_orientations[agent_id] = (ori - 1) % 4
            reward += -1

        elif action == 3:  # rotate right
            self.agent_orientations[agent_id] = (ori + 1) % 4
            reward += -1

        elif action == 4:  # pick
            if self.agent_carrying[agent_id] is None:
                for i, (req_id, src, _, picked) in enumerate(self.requests):
                    if src == (fx, fy) and not picked:
                        self.agent_carrying[agent_id] = req_id  # Store request ID, not index
                        self.requests[i] = (req_id, src, self.requests[i][2], True)  # Update picked status
                        for req in self.requests_static:
                            if req[0] == req_id:
                                req[-1] = True
                                break
                        reward += 50
                        break
                else:
                    reward += -4
            # if it has already picked or there is no package, reward -4
            else:
                reward += -4

        elif action == 5:  # place
            if self.agent_carrying[agent_id] is not None:
                req_id = self.agent_carrying[agent_id]
                # Find the request with this ID
                for i, (r_id, _, dest, _) in enumerate(self.requests):
                    if r_id == req_id:
                        if (fx, fy) == dest:
                            reward += 100
                            self.requests.pop(i)  # Remove the request
                            for req in self.requests_static:
                                if req[0] == r_id:
                                    req[0] = -1
                                    req[1] = np.ones(2, dtype=np.int8) * -1
                                    req[2] = np.ones(2, dtype=np.int8) * -1
                                    req[3] = False
                                    break
                            print(f"Agent {agent_id} delivered request {req_id} to {dest}.")
                            self.agent_carrying[agent_id] = None
                        else:
                            reward += -4
                        break
                else:
                    reward += -4

            else:
                reward += -4
        elif action == 6:  # idle
            reward += -1

        if self.agent_carrying[agent_id] is not None and action in [0, 1]:
            req_id = self.agent_carrying[agent_id]
            for r_id, _, dest, _ in self.requests:
                if r_id == req_id and prev_distance is not None:
                    new_distance = abs(new_pos[0] - dest[0]) + abs(new_pos[1] - dest[1])
                    if new_distance < prev_distance:
                        reward += 3
                    elif new_distance > prev_distance:
                        reward += -3
                    else:
                        reward += -1
                    break

        return reward

    def _obs_centralized(self):
        """Return centralized observations as a structured dictionary."""
        # Create the grid observation
        grid_obs = self.grid.copy()
        
        # Create the agents observation
        agents_obs = []
        for i, ((x, y), ori) in enumerate(zip(self.agent_positions, self.agent_orientations)):
            carrying = -1 if self.agent_carrying[i] is None else self.agent_carrying[i]
            agents_obs.append({
                'position': np.array([x, y], dtype=np.int8),
                'orientation': ori,
                'carrying': np.array([carrying], dtype=np.int8)
            })
        
        # Create the requests observation
        requests_obs = []
        for i in range(self.max_requests):
            if i < len(self.requests):
                req_id, src, dest, is_picked = self.requests[i]
                requests_obs.append({
                    'id': req_id,
                    'source': np.array(src, dtype=np.int8),
                    'destination': np.array(dest, dtype=np.int8),
                    'picked': int(is_picked),
                })
            else:
                # Placeholder for unused request slots
                requests_obs.append({
                    'id': -1,
                    'source': np.ones(2, dtype=np.int8) * -1,
                    'destination': np.ones(2, dtype=np.int8) * -1,
                    'picked': -1,
                })
        
        return {
            'grid': grid_obs,
            'agents': tuple(agents_obs),
            'requests': tuple(requests_obs)
        }

    def _obs_decentralized(self):
        """Return decentralized observations for each agent with full observability."""
        # Create the grid observation (shared by all agents)
        grid_obs = self.grid.copy()
        
        # Create per-agent observations
        agent_observations = []
        
        for i, ((x, y), ori) in enumerate(zip(self.agent_positions, self.agent_orientations)):
            # Get carrying status
            carrying = -1 if self.agent_carrying[i] is None else self.agent_carrying[i]
            
            # Create agent-specific observation
            agent_obs = {
                'grid': grid_obs,
                'position': np.array([x, y], dtype=np.int8),
                'orientation': ori,
                'carrying': np.array([carrying], dtype=np.int8),
                'requests': []
            }
            
            # Add all requests to the observation
            for j in range(self.max_requests):
                if j < len(self.requests):
                    req_id, src, dest, is_picked = self.requests[j]
                    agent_obs['requests'].append({
                        'id': req_id,
                        'source': np.array(src, dtype=np.int8),
                        'destination': np.array(dest, dtype=np.int8),
                        'picked': int(is_picked),
                    })
                else:
                    # Placeholder for unused request slots
                    agent_obs['requests'].append({
                        'id': -1,
                        'source': np.ones(2, dtype=np.int8) * -1,
                        'destination': np.ones(2, dtype=np.int8) * -1,
                        'picked': -1,
                    })

            agent_observations.append(agent_obs)
        
        return agent_observations

    def _obs_decentralized_partial_observability(self):
        """Return decentralized observations for each agent with partial observability (5x5 grid)."""
        # Create per-agent observations
        agent_observations = []
        
        for i, ((x, y), ori) in enumerate(zip(self.agent_positions, self.agent_orientations)):
            # Create a 5x5 grid centered on the agent
            # -1 represents walls/out of bounds
            partial_grid = np.full((5, 5), -1, dtype=np.int8)
            
            # Fill in the visible part of the grid
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                        partial_grid[dx+2, dy+2] = self.grid[nx, ny]
                        
                        # Mark other agents' positions
                        for j, (ax, ay) in enumerate(self.agent_positions):
                            if j != i and (ax, ay) == (nx, ny):
                                partial_grid[dx+2, dy+2] = 3  # Use 3 to represent other agents
            
            # Get carrying status
            carrying = -1 if self.agent_carrying[i] is None else self.agent_carrying[i]
            
            # Create agent-specific observation
            agent_obs = {
                'grid': partial_grid,
                'position': np.array([x, y], dtype=np.int8),
                'orientation': ori,
                'carrying': np.array([carrying], dtype=np.int8),
                'requests': []
            }
            
            # Add all requests to the observation
            # For partial observability, we only include requests that are visible
            for j in range(self.max_requests):
                if j < len(self.requests):
                    req_id, src, dest, is_picked = self.requests[j]
                    
                    agent_obs['requests'].append({
                        'id': req_id,
                        'source': np.array(src, dtype=np.int8),
                        'destination': np.array(dest, dtype=np.int8),
                        'picked': int(is_picked),
                    })
                else:
                    # Placeholder for unused request slots
                    agent_obs['requests'].append({
                        'id': -1,
                        'source': np.ones(2, dtype=np.int8) * -1,
                        'destination': np.ones(2, dtype=np.int8) * -1,
                        'picked': -1,
                    })

            agent_observations.append(agent_obs)
        
        return agent_observations

    def _dict_to_vector(self, obs_dict):
        """Convert a structured observation dictionary to a flat vector.
        
        Args:
            obs_dict: The structured observation dictionary
            normalize: If True, normalize values to [0, 1] range (preserving -1 values)
        """
        max_pos = max(self.grid_size)
        max_grid_val = 3  # 0: empty, 1: shelf, 2: destination, 3: agent
        
        if self.observation_type == 'centralized':
            # Normalize grid values if requested
            if self.normalize:
                grid_flat = obs_dict['grid'].flatten() / max_grid_val
            else:
                grid_flat = obs_dict['grid'].flatten()
            
            vector = grid_flat
            
            # Add agent information
            for agent in obs_dict['agents']:
                # Normalize position
                if self.normalize:
                    position = agent['position'] / max_pos
                    orientation = np.array([agent['orientation'] / 3.0], dtype=np.float32)
                    
                    # Keep -1 as -1, normalize others
                    carrying = agent['carrying'].astype(np.float32)
                    carrying_mask = carrying >= 0
                    carrying[carrying_mask] = carrying[carrying_mask] / self.max_requests
                else:
                    position = agent['position']
                    orientation = np.array([agent['orientation']], dtype=np.int8)
                    carrying = agent['carrying']
                
                vector = np.concatenate([vector, position, orientation, carrying])
            
            # Add request information
            for request in obs_dict['requests']:
                if self.normalize:
                    # Normalize request ID, preserving -1
                    req_id = request['id']
                    if req_id == -1:
                        req_id_ = np.array([-1.0], dtype=np.float32)
                    else:
                        req_id_ = np.array([float(req_id) / self.goal_deliveries], dtype=np.float32)
                    
                    # Normalize source coordinates, preserving -1
                    source = request['source'].astype(np.float32)
                    source_mask = source >= 0
                    source[source_mask] = source[source_mask] / max_pos
                    
                    # Normalize destination coordinates, preserving -1
                    destination = request['destination'].astype(np.float32)
                    dest_mask = destination >= 0
                    destination[dest_mask] = destination[dest_mask] / max_pos
                    
                    # Normalize picked status, preserving -1
                    picked_val = request['picked']
                    if picked_val == -1:
                        picked = np.array([-1.0], dtype=np.float32)
                    else:
                        picked = np.array([float(picked_val)], dtype=np.float32)
                else:
                    req_id_ = np.array([request['id']], dtype=np.int8)
                    source = request['source']
                    destination = request['destination']
                    picked = np.array([request['picked']], dtype=np.int8)
                
                vector = np.concatenate([
                    vector,
                    req_id_,
                    source,
                    destination,
                    picked
                ])
            
            return vector
        
        elif self.observation_type in ['decentralized', 'decentralized_partial_observability']:
            # Process each agent's observation
            agent_vectors = []
            
            for agent_obs in obs_dict:
                # Normalize grid values if requested
                if self.normalize:
                    # Preserve -1 values, normalize others
                    grid_flat = agent_obs['grid'].flatten().astype(np.float32)
                    grid_mask = grid_flat >= 0
                    grid_flat[grid_mask] = grid_flat[grid_mask] / max_grid_val
                else:
                    grid_flat = agent_obs['grid'].flatten()
                
                vector = grid_flat
                
                # Add agent information
                if self.normalize:
                    position = agent_obs['position'] / max_pos
                    orientation = np.array([agent_obs['orientation'] / 3.0], dtype=np.float32)
                    
                    # Keep -1 as -1, normalize others
                    carrying = agent_obs['carrying'].astype(np.float32)
                    carrying_mask = carrying >= 0
                    carrying[carrying_mask] = carrying[carrying_mask] / self.max_requests
                else:
                    position = agent_obs['position']
                    orientation = np.array([agent_obs['orientation']], dtype=np.int8)
                    carrying = agent_obs['carrying']
                
                vector = np.concatenate([vector, position, orientation, carrying])
                
                # Add request information
                for request in agent_obs['requests']:
                    if self.normalize:
                        # Normalize request ID, preserving -1
                        req_id = request['id']
                        if req_id == -1:
                            req_id_ = np.array([-1.0], dtype=np.float32)
                        else:
                            req_id_ = np.array([float(req_id) / self.goal_deliveries], dtype=np.float32)
                        
                        # Normalize source coordinates, preserving -1
                        source = request['source'].astype(np.float32)
                        source_mask = source >= 0
                        source[source_mask] = source[source_mask] / max_pos
                        
                        # Normalize destination coordinates, preserving -1
                        destination = request['destination'].astype(np.float32)
                        dest_mask = destination >= 0
                        destination[dest_mask] = destination[dest_mask] / max_pos
                        
                        # Normalize picked status, preserving -1
                        picked_val = request['picked']
                        if picked_val == -1:
                            picked = np.array([-1.0], dtype=np.float32)
                        else:
                            picked = np.array([float(picked_val)], dtype=np.float32)
                    else:
                        req_id_ = np.array([request['id']], dtype=np.int8)
                        source = request['source']
                        destination = request['destination']
                        picked = np.array([request['picked']], dtype=np.int8)
                    
                    vector = np.concatenate([
                        vector,
                        req_id_,
                        source,
                        destination,
                        picked
                    ])
                
                agent_vectors.append(vector)
            
            return np.array(agent_vectors)

    def _get_obs(self):
        """Return the appropriate observation based on the environment configuration."""
        # Get structured observation
        if self.observation_type == 'centralized':
            structured_obs = self._obs_centralized()
        elif self.observation_type == 'decentralized':
            structured_obs = self._obs_decentralized()
        elif self.observation_type == 'decentralized_partial_observability':
            structured_obs = self._obs_decentralized_partial_observability()
        else:
            raise ValueError(f"Unknown observation type: {self.observation_type}")
        
        # Convert to vector
        vector_obs = self._dict_to_vector(structured_obs)
        
        return vector_obs, structured_obs

    def _get_info(self):
        """Return additional information about the environment state."""
        return {
            'num_requests': len(self.requests),
            'num_picked_requests': sum(1 for _, _, picked in self.requests if picked),
            'num_agents_carrying': sum(1 for c in self.agent_carrying if c is not None),
            'steps_taken': self.steps_taken,
            'deliveries_completed': self.deliveries_completed,
            'agent_positions': self.agent_positions.copy(),
            'agent_orientations': self.agent_orientations.copy(),
            'agent_carrying': self.agent_carrying.copy(),
            'requests': self.requests.copy(),
            'grid': self.grid.copy(),
            'static_requests': self.requests_static.copy()
        }
    
    def render(self):
        # Create figure and axes if they don't exist
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            plt.ion()  # Turn on interactive mode
        
        # Clear the current axes for redrawing
        self.ax.clear()
        
        # Set up the plot
        self.ax.set_xlim(0, self.grid_size[1])
        self.ax.set_ylim(0, self.grid_size[0])
        self.ax.set_xticks(np.arange(self.grid_size[1] + 1))
        self.ax.set_yticks(np.arange(self.grid_size[0] + 1))
        self.ax.grid(True)

        colors = {0: 'white', 1: 'gray', 2: 'green', 3: 'white', 4: 'gray'}
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.ax.add_patch(patches.Rectangle((y, self.grid_size[0] - 1 - x), 1, 1,
                                               facecolor=colors[self.grid[x, y]], edgecolor='black'))

        # Direction vectors for each orientation (dx, dy)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left
        
        for i, ((x, y), ori) in enumerate(zip(self.agent_positions, self.agent_orientations)):
            # Convert grid coordinates to plot coordinates
            plot_x = y + 0.5
            plot_y = self.grid_size[0] - 1 - x + 0.5
            
            # Set agent color based on carrying status (green if carrying, red if not)
            agent_color = 'green' if self.agent_carrying[i] is not None else 'red'
            
            # Draw agent as an arrow
            dx, dy = directions[ori]
            arrow_length = 0.4
            self.ax.arrow(plot_x - dx * arrow_length/2, plot_y - dy * arrow_length/2, 
                     dx * arrow_length, dy * arrow_length, 
                     head_width=0.2, head_length=0.2, fc=agent_color, ec=agent_color, 
                     width=0.1, length_includes_head=True)
            
            # Add agent number
            text_offset_x = -0.2 if dx == 0 else (0.2 if dx > 0 else -0.3)
            text_offset_y = -0.2 if dy == 0 else (0.2 if dy > 0 else -0.3)
            self.ax.text(plot_x + text_offset_x, plot_y + text_offset_y, str(i), 
                    color='white', ha='center', va='center', fontsize=9, 
                    fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, pad=1))

        # Draw requests
        for req_id, src, dest, picked in self.requests:
            if not picked:
                # Source location with a package
                self.ax.add_patch(patches.Rectangle((src[1] + 0.25, self.grid_size[0] - 1 - src[0] + 0.25), 
                                              0.5, 0.5, color='brown'))
            else:
                # Destination marker
                self.ax.add_patch(patches.Rectangle((dest[1] + 0.25, self.grid_size[0] - 1 - dest[0] + 0.25), 
                                              0.5, 0.5, color='green', alpha=0.5))

        self.ax.set_aspect('equal', adjustable='box')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to allow the GUI to update

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

# if __name__ == "__main__":
#     env = WarehouseEnv(grid_size=(15, 9), num_agents=3, observation_type='decentralized_partial_observability')
#     obs, _ = env.reset()
#     terminated = [False] * env.num_agents
#     truncated = [False] * env.num_agents
#     try:
#         while not any(terminated) and not any(truncated):
#             env.render()
#             # print(env.grid)
#             time.sleep(5)
#             actions = env.action_space.sample()
#             obs, reward, terminated, truncated, _ = env.step(actions)
#             # print(f"Reward: {reward}")
#     finally:
#         env.close()  # Make sure to close the environment