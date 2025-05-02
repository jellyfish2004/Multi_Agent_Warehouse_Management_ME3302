import random
import torch
import numpy as np
from collections import deque, namedtuple
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
from warehouse_env import WarehouseEnv
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hrl_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BUFFER_SIZE = int(2e5)    # Increased replay buffer size
BATCH_SIZE = 128          # Larger batch size for more stable learning
GAMMA = 0.99              # Keep discount factor
LR = 1e-4                 # Lower learning rate for more stable learning
UPDATE_EVERY = 10         # Update target network more frequently
OPTION_DURATION = 15      # Longer option duration to allow for more coherent behavior
PRIORITIZED_REPLAY = True # Enable prioritized experience replay
DELIVERY_BONUS_SAMPLES = 10  # Number of times to add successful delivery experiences to buffer
REWARD_SCALE = 0.1        # Scale rewards to help with learning stability

class QNetwork1(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork1, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Option Network: Takes state, destination, and grid to output primitive actions
class OptionNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(OptionNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> primitive action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class HRLAgent():
    def __init__(self, state_size, action_size, num_options, seed):
        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.num_options = num_options  # Number of options (max_requests + n_destinations)
        self.seed = random.seed(seed)

        ''' Q-Networks '''
        # High-level policy (option selection)
        self.option_qnetwork_local = QNetwork1(state_size, num_options, seed).to(device)
        self.option_qnetwork_target = QNetwork1(state_size, num_options, seed).to(device)
        self.option_optimizer = optim.Adam(self.option_qnetwork_local.parameters(), lr=LR)

        # Low-level policy (primitive action selection for each option)
        self.primitive_qnetwork = OptionNetwork(state_size + 4, action_size, seed).to(device)  # +4 for destination coordinates and option type
        self.primitive_optimizer = optim.Adam(self.primitive_qnetwork.parameters(), lr=LR)

        ''' Replay memory '''
        self.option_memory = ReplayBuffer(num_options, BUFFER_SIZE, BATCH_SIZE, seed)
        self.primitive_memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        ''' Initialize time step (for updating every UPDATE_EVERY steps) '''
        self.t_step = 0

        ''' Option tracking '''
        self.current_options = [None] * 10  # Assuming max 10 agents, will store (option_id, target_location, steps_remaining)
        self.option_destinations = {}  # Maps request_id to destination coordinates

        # Store the last augmented state for each agent
        self.last_augmented_states = [None] * 10

    def step(self, state, action, reward, next_state, done, agent_id=0, is_option=False, 
             augmented_state=None, next_augmented_state=None, delivery_success=False):
        ''' Save experience in replay memory '''
        # Scale rewards for better learning stability
        scaled_reward = reward * REWARD_SCALE
        
        if is_option:
            self.option_memory.add(state, action, scaled_reward, next_state, done)
            # If this was a successful delivery, add it multiple times to prioritize this experience
            if delivery_success:
                for _ in range(DELIVERY_BONUS_SAMPLES):
                    self.option_memory.add(state, action, scaled_reward, next_state, done)
        else:
            # Use the augmented states for primitive actions
            if augmented_state is not None and next_augmented_state is not None:
                self.primitive_memory.add(augmented_state, action, scaled_reward, next_augmented_state, done)
                # If this was a successful delivery, add it multiple times
                if delivery_success:
                    for _ in range(DELIVERY_BONUS_SAMPLES):
                        self.primitive_memory.add(augmented_state, action, scaled_reward, next_augmented_state, done)
            else:
                # If augmented states aren't provided, create default ones
                default_augmented_state = np.concatenate([
                    state, 
                    np.array([0, 0], dtype=np.float32),  # Default destination
                    np.array([1, 0], dtype=np.float32)   # Default option type (source)
                ])
                
                default_next_augmented_state = np.concatenate([
                    next_state, 
                    np.array([0, 0], dtype=np.float32),  # Default destination
                    np.array([1, 0], dtype=np.float32)   # Default option type (source)
                ])
                
                self.primitive_memory.add(default_augmented_state, action, scaled_reward, default_next_augmented_state, done)

        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.option_memory) >= BATCH_SIZE:
            experiences = self.option_memory.sample()
            self.learn(experiences, GAMMA, is_option=True)
            
        if len(self.primitive_memory) >= BATCH_SIZE:
            experiences = self.primitive_memory.sample()
            self.learn(experiences, GAMMA, is_option=False)

        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            self.option_qnetwork_target.load_state_dict(self.option_qnetwork_local.state_dict())

    def select_option(self, state, eps=0.):
        """Select an option using epsilon-greedy policy"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.option_qnetwork_local.eval()
        with torch.no_grad():
            option_values = self.option_qnetwork_local(state)
        self.option_qnetwork_local.train()

        # Epsilon-greedy option selection
        if random.random() > eps:
            return np.argmax(option_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.num_options))

    def select_primitive_action(self, state, destination, option_type, eps=0., agent_id=0):
        """Select a primitive action for the current option"""
        # Augment state with destination and option type
        dest_x, dest_y = destination
        option_type_onehot = [1, 0] if option_type == "source" else [0, 1]  # source or destination
        
        # Create augmented state: original state + destination coordinates + option type
        augmented_state = np.concatenate([
            state, 
            np.array([dest_x, dest_y], dtype=np.float32),
            np.array(option_type_onehot, dtype=np.float32)
        ])
        
        # Store the augmented state for this agent
        self.last_augmented_states[agent_id] = augmented_state
        
        augmented_state_tensor = torch.from_numpy(augmented_state).float().unsqueeze(0).to(device)
        
        self.primitive_qnetwork.eval()
        with torch.no_grad():
            action_values = self.primitive_qnetwork(augmented_state_tensor)
        self.primitive_qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), augmented_state
        else:
            return random.choice(np.arange(self.action_size)), augmented_state

    def act(self, state, agent_id, requests, destinations, eps=0.):
        """
        Decide whether to use an option or a primitive action
        
        Args:
            state: The current state
            agent_id: The ID of the agent
            requests: List of active requests
            destinations: List of destination coordinates
            eps: Epsilon for exploration
        """
        # Check if agent is currently executing an option
        if self.current_options[agent_id] is not None:
            option_id, target, steps_remaining = self.current_options[agent_id]
            
            # If option has steps remaining, continue with it
            if steps_remaining > 0:
                # Determine if this is a source or destination option
                option_type = "source" if option_id < len(requests) else "destination"
                
                # Select primitive action based on current state and target
                primitive_action, augmented_state = self.select_primitive_action(state, target, option_type, eps, agent_id)
                
                # Update steps remaining
                self.current_options[agent_id] = (option_id, target, steps_remaining - 1)
                
                return primitive_action, True, augmented_state  # True indicates this is from an option
            
        # Either no current option or option completed, select a new option or primitive action
        option = self.select_option(state, eps)
        
        # Determine target location based on option
        if option < len(requests):
            # Option is to go to a request source
            if option < len(requests) and len(requests) > 0:
                req_id, src, dest, picked = requests[option]
                if not picked:  # Only valid if not already picked
                    self.current_options[agent_id] = (option, src, OPTION_DURATION)
                    self.option_destinations[req_id] = dest  # Store destination for later
                    
                    # Select primitive action for this option
                    primitive_action, augmented_state = self.select_primitive_action(state, src, "source", eps, agent_id)
                    return primitive_action, True, augmented_state
        else:
            # Option is to go to a destination
            dest_idx = option - len(requests)
            if dest_idx < len(destinations) and len(destinations) > 0:
                self.current_options[agent_id] = (option, destinations[dest_idx], OPTION_DURATION)
                
                # Select primitive action for this option
                primitive_action, augmented_state = self.select_primitive_action(state, destinations[dest_idx], "destination", eps, agent_id)
                return primitive_action, True, augmented_state
        
        # If we get here, either option selection failed or we need a primitive action
        # Create a properly sized augmented state for the primitive network
        # Use a default destination (0,0) and option type "source"
        augmented_state = np.concatenate([
            state, 
            np.array([0, 0], dtype=np.float32),  # Default destination
            np.array([1, 0], dtype=np.float32)   # Default option type (source)
        ])
        
        # Store the augmented state for this agent
        self.last_augmented_states[agent_id] = augmented_state
        
        augmented_state_tensor = torch.from_numpy(augmented_state).float().unsqueeze(0).to(device)
        
        self.primitive_qnetwork.eval()
        with torch.no_grad():
            action_values = self.primitive_qnetwork(augmented_state_tensor)
        self.primitive_qnetwork.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()), False, augmented_state
        else:
            return random.choice(np.arange(self.action_size)), False, augmented_state

    def learn(self, experiences, gamma, is_option=False):
        """Update value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        if is_option:
            # Option Q-learning update
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.option_qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            
            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            
            # Get expected Q values from local model
            Q_expected = self.option_qnetwork_local(states).gather(1, actions)
            
            # Compute loss
            loss = F.mse_loss(Q_expected, Q_targets)
            
            # Minimize the loss
            self.option_optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            for param in self.option_qnetwork_local.parameters():
                param.grad.data.clamp_(-1, 1)
                
            self.option_optimizer.step()
            
            return loss.item()
        else:
            # Primitive action Q-learning update
            # For primitive actions, we don't have a target network, just update directly
            Q_targets_next = self.primitive_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_expected = self.primitive_qnetwork(states).gather(1, actions)
            
            loss = F.mse_loss(Q_expected, Q_targets)
            
            self.primitive_optimizer.zero_grad()
            loss.backward()
            
            for param in self.primitive_qnetwork.parameters():
                param.grad.data.clamp_(-1, 1)
            
            self.primitive_optimizer.step()
            
            return loss.item()

    def reset_options(self):
        """Reset all options at the start of a new episode"""
        self.current_options = [None] * 10
        self.option_destinations = {}
        self.last_augmented_states = [None] * 10


def train_hrl(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, render_every=100):
    """Train the hierarchical reinforcement learning agent."""
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores for calculating moving average
    eps = eps_start                    # initialize epsilon
    
    option_losses = []
    primitive_losses = []
    deliveries_per_episode = []
    
    logger.info("Starting training...")
    
    # Main training loop with tqdm progress bar
    for i_episode in tqdm(range(1, n_episodes+1), desc="Training Episodes"):
        obs_vector, obs_structured = env.reset()
        agent.reset_options()  # Reset options at the start of each episode
        
        episode_score = 0
        episode_option_loss = 0
        episode_primitive_loss = 0
        option_loss_count = 0
        primitive_loss_count = 0
        episode_deliveries = 0
        
        # Keep track of previous requests to detect deliveries
        prev_requests = env.requests.copy()
        
        # Episode loop with tqdm progress bar for steps
        for t in tqdm(range(max_t), desc=f"Episode {i_episode} Steps", leave=False):
            actions = []
            is_option_flags = []
            augmented_states = []
            
            # Get actions for each agent
            for agent_id in range(env.num_agents):
                # Extract agent-specific observation
                if env.observation_type == 'centralized':
                    agent_state = obs_vector
                else:
                    agent_state = obs_vector[agent_id]
                
                # Get destinations from environment
                destinations = env.destinations
                
                # Get action (either option or primitive)
                action, is_option, augmented_state = agent.act(agent_state, agent_id, env.requests, destinations, eps)
                actions.append(action)
                is_option_flags.append(is_option)
                augmented_states.append(augmented_state)
            
            # Take actions in environment
            next_obs_vector, rewards, terminated, truncated, next_obs_structured = env.step(actions)
            
            # Check for deliveries by comparing previous and current requests
            deliveries = []
            for agent_id in range(env.num_agents):
                delivery_made, delivery_dest = track_delivery(
                    env, prev_requests, env.requests, agent_id, env.agent_carrying
                )
                deliveries.append(delivery_made)
                if delivery_made:
                    episode_deliveries += 1
                    logger.info(f"Agent {agent_id} made a delivery!")
            
            # Update previous requests
            prev_requests = env.requests.copy()
            
            # Process rewards and update agent
            for agent_id in range(env.num_agents):
                # Extract agent-specific observations
                if env.observation_type == 'centralized':
                    agent_state = obs_vector
                    agent_next_state = next_obs_vector
                    agent_reward = rewards  # In centralized mode, reward is already averaged
                    agent_done = terminated or truncated
                else:
                    agent_state = obs_vector[agent_id]
                    agent_next_state = next_obs_vector[agent_id]
                    agent_reward = rewards[agent_id]
                    agent_done = terminated[agent_id] or truncated[agent_id]
                
                # Create next augmented state
                if env.observation_type == 'centralized':
                    next_augmented_state = np.concatenate([
                        agent_next_state, 
                        np.array([0, 0], dtype=np.float32),  # Default destination
                        np.array([1, 0], dtype=np.float32)   # Default option type
                    ])
                else:
                    next_augmented_state = np.concatenate([
                        agent_next_state, 
                        np.array([0, 0], dtype=np.float32),  # Default destination
                        np.array([1, 0], dtype=np.float32)   # Default option type
                    ])
                
                # Update agent with experience
                loss = agent.step(
                    agent_state, 
                    actions[agent_id], 
                    agent_reward, 
                    agent_next_state, 
                    agent_done,
                    agent_id=agent_id,
                    is_option=is_option_flags[agent_id],
                    augmented_state=augmented_states[agent_id],
                    next_augmented_state=next_augmented_state,
                    delivery_success=deliveries[agent_id]
                )
                
                # Track losses
                if is_option_flags[agent_id]:
                    if loss is not None:
                        episode_option_loss += loss
                        option_loss_count += 1
                else:
                    if loss is not None:
                        episode_primitive_loss += loss
                        primitive_loss_count += 1
                
                # Add reward to score
                if env.observation_type == 'centralized':
                    episode_score = rewards  # In centralized mode, reward is already the total
                    break  # Only need to process once for centralized
                else:
                    episode_score += agent_reward / env.num_agents  # Average reward across agents
            
            # Update observations for next step
            obs_vector = next_obs_vector
            obs_structured = next_obs_structured
            
            # Check if episode is done
            if env.observation_type == 'centralized':
                if terminated or truncated:
                    break
            else:
                if all(terminated) or all(truncated):
                    break
            
            # Render environment occasionally
            if i_episode % render_every == 0 and t % 10 == 0:
                env.render()
        
        # Calculate average losses for this episode
        avg_option_loss = episode_option_loss / max(1, option_loss_count) if option_loss_count > 0 else 0
        avg_primitive_loss = episode_primitive_loss / max(1, primitive_loss_count) if primitive_loss_count > 0 else 0
        option_losses.append(avg_option_loss)
        primitive_losses.append(avg_primitive_loss)
        deliveries_per_episode.append(episode_deliveries)
        
        # Save score and update epsilon
        scores_window.append(episode_score)
        scores.append(episode_score)
        eps = max(eps_end, eps_decay*eps)
        
        # Log progress
        logger.info(f"Episode {i_episode}/{n_episodes} - Score: {episode_score:.2f} - Deliveries: {episode_deliveries}")
        logger.info(f"Avg Score: {np.mean(scores_window):.2f} - Epsilon: {eps:.2f}")
        logger.info(f"Option Loss: {avg_option_loss:.4f} - Primitive Loss: {avg_primitive_loss:.4f}")
        
        if i_episode % 100 == 0:
            logger.info(f"Episode {i_episode}/{n_episodes} - Avg Score (100 ep): {np.mean(scores_window):.2f}")
            logger.info(f"Avg Deliveries (100 ep): {np.mean(deliveries_per_episode[-100:]):.2f}")
            
            # Save model weights periodically
            torch.save(agent.option_qnetwork_local.state_dict(), f'option_qnetwork_checkpoint_{i_episode}.pth')
            torch.save(agent.primitive_qnetwork.state_dict(), f'primitive_qnetwork_checkpoint_{i_episode}.pth')
            
            # Plot intermediate results
            plot_scores(scores, filename=f'hrl_scores_episode_{i_episode}.png')
            plot_losses(option_losses, primitive_losses, filename=f'hrl_losses_episode_{i_episode}.png')
            plot_deliveries(deliveries_per_episode, filename=f'hrl_deliveries_episode_{i_episode}.png')
    
    # Save final model weights
    torch.save(agent.option_qnetwork_local.state_dict(), 'option_qnetwork_final.pth')
    torch.save(agent.primitive_qnetwork.state_dict(), 'primitive_qnetwork_final.pth')
    
    return scores, option_losses, primitive_losses, deliveries_per_episode

def plot_scores(scores, filename='hrl_scores.png'):
    """Plot the scores."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('HRL Training Scores')
    
    # Add moving average
    if len(scores) >= 100:
        moving_avg = np.convolve(scores, np.ones((100,))/100, mode='valid')
        plt.plot(np.arange(len(moving_avg)), moving_avg, 'r-')
        plt.legend(['Score', 'Moving Avg (100 episodes)'])
    else:
        plt.legend(['Score'])
    
    plt.savefig(filename)
    plt.close()

def plot_losses(option_losses, primitive_losses, filename='hrl_losses.png'):
    """Plot the losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(option_losses)), option_losses, 'b-')
    plt.plot(np.arange(len(primitive_losses)), primitive_losses, 'g-')
    plt.ylabel('Loss')
    plt.xlabel('Episode #')
    plt.title('HRL Training Losses')
    plt.legend(['Option Network Loss', 'Primitive Network Loss'])
    plt.yscale('log')  # Use log scale for better visualization
    plt.savefig(filename)
    plt.close()

def plot_deliveries(deliveries, filename='hrl_deliveries.png'):
    """Plot the number of deliveries per episode."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(deliveries)), deliveries)
    plt.ylabel('Deliveries')
    plt.xlabel('Episode #')
    plt.title('Deliveries per Episode')
    
    # Add moving average
    if len(deliveries) >= 100:
        moving_avg = np.convolve(deliveries, np.ones((100,))/100, mode='valid')
        plt.plot(np.arange(len(moving_avg)), moving_avg, 'r-')
        plt.legend(['Deliveries', 'Moving Avg (100 episodes)'])
    else:
        plt.legend(['Deliveries'])
    
    plt.savefig(filename)
    plt.close()

def track_delivery(env, prev_requests, curr_requests, agent_id, agent_carrying):
    """Track if a delivery was completed and return the delivery details if so."""
    if len(prev_requests) > len(curr_requests) and agent_carrying[agent_id] is not None:
        # A request was completed (removed from the list)
        for prev_req in prev_requests:
            req_id, _, dest, _ = prev_req
            if req_id == agent_carrying[agent_id]:
                # This was the request that was delivered
                return True, dest
    return False, None

if __name__ == "__main__":
    # Initialize environment with original settings
    env = WarehouseEnv(
        grid_size=(15, 10),
        num_agents=3,  # Using 3 agents
        num_shelves=10,
        shelf_spacing=2,
        shelf_length=4,
        poisson_lambda=0.1,  # Original request rate
        observation_type='decentralized',  # Using decentralized observations
        normalize=True,
        max_requests=5,  # Original max requests
        goal_deliveries=10  # Original goal deliveries
    )
    
    # Get state and action dimensions
    if env.observation_type == 'centralized':
        state_size = env.observation_space.shape[0]
    else:
        state_size = env.observation_space.shape[1]  # Each agent's observation size
    
    action_size = 7  # Number of primitive actions
    
    # Calculate number of options: max_requests + num_destinations
    num_options = env.max_requests + len(env.destinations)  # Use original calculation
    
    logger.info(f"Environment initialized with {env.num_agents} agents")
    logger.info(f"State size: {state_size}, Action size: {action_size}, Options: {num_options}")
    
    # Initialize agent
    agent = HRLAgent(
        state_size=state_size,
        action_size=action_size,
        num_options=num_options,
        seed=0
    )
    
    # Train agent
    begin_time = datetime.datetime.now()
    logger.info(f"Training started at: {begin_time}")
    
    scores, option_losses, primitive_losses, deliveries = train_hrl(
        env=env,
        agent=agent,
        n_episodes=2000,
        max_t=1000,
        eps_start=1.0,
        eps_end=0.01,  # Original epsilon end
        eps_decay=0.995,
        render_every=100
    )
    
    time_taken = datetime.datetime.now() - begin_time
    logger.info(f"Training completed in: {time_taken}")
    
    # Plot final results
    plot_scores(scores)
    plot_losses(option_losses, primitive_losses)
    plot_deliveries(deliveries)
    
    # Close environment
    env.close()



