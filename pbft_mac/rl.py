import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Tuple, List, Optional
from collections import defaultdict, deque
import random

from .core import (
    Config,
    ScenarioParams,
    CSMAParams,
    TDMAParams,
    PBFTResult,
    ResultsAccumulator,
    init_positions,
    init_velocities,
    update_positions,
    extract_context,
    csma_backoff_delay,
    next_tdma_start,
    link_delay,
    count_transmissions,
    calculate_energy,
    calculate_throughput,
)

# ============================================================================
# GLOBAL SETUP
# ============================================================================

# Device selection for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[RL] Using device: {device}")


# ============================================================================
# HELPER FUNCTIONS: STATE REPRESENTATION
# ============================================================================

def discretize_context(ctx: Dict, n_bins: int = 10) -> Tuple:
    """
    Discretize continuous context features into bins for tabular Q-learning.
    
    Args:
        ctx: Context dictionary with keys 'dCenters', 'meanD', 'estPER', 
             'N', 'if_prob', 'bgLoad'
        n_bins: Number of bins for discretization (default: 10)
    
    Returns:
        Tuple of discretized state values (d_centers_bin, mean_d_bin, 
        per_bin, n_nodes, if_prob_bin, bg_load_bin)
    """
    # Discretize distance to center (0-1000m range)
    d_centers_bin = min(n_bins - 1, int(ctx['dCenters'] / 100.0))
    
    # Discretize mean distance (0-1000m range)
    mean_d_bin = min(n_bins - 1, int(ctx['meanD'] / 100.0))
    
    # Discretize PER (0-1 range)
    per_bin = min(n_bins - 1, int(ctx['estPER'] * n_bins))
    
    # Node count (direct value, typically 4-10)
    n_nodes = int(ctx['N'])
    
    # Interference probability (0-1 range)
    if_prob_bin = min(n_bins - 1, int(ctx['if_prob'] * n_bins))
    
    # Background load (0-1 range)
    bg_load_bin = min(n_bins - 1, int(ctx['bgLoad'] * n_bins))
    
    return (d_centers_bin, mean_d_bin, per_bin, n_nodes, if_prob_bin, bg_load_bin)


def context_to_state_vector(ctx: Dict) -> np.ndarray:
    """
    Convert context dictionary to continuous state vector for neural network agents.
    
    Args:
        ctx: Context dictionary with keys 'dCenters', 'meanD', 'estPER', 
             'N', 'if_prob', 'bgLoad'
    
    Returns:
        Normalized numpy array of shape (6,) with features in [0, 1] range
    """
    return np.array([
        ctx['dCenters'] / 1000.0,    # Normalize to [0, 1] (max 1000m)
        ctx['meanD'] / 1000.0,        # Normalize to [0, 1] (max 1000m)
        ctx['estPER'],                # Already in [0, 1]
        ctx['N'] / 10.0,              # Normalize assuming max 10 nodes
        ctx['if_prob'],               # Already in [0, 1]
        ctx['bgLoad']                 # Already in [0, 1]
    ], dtype=np.float32)


# ============================================================================
# PBFT SIMULATION CORE
# ============================================================================

def pbft_round(
    mac_type: str,
    primary: int,
    pos0: np.ndarray,
    N: int,
    t0: float,
    eff: Dict,
    scn: ScenarioParams,
    cs: CSMAParams,
    tdma: TDMAParams,
    cfg: Config
) -> PBFTResult:
    """
    Execute one PBFT consensus round with specified MAC protocol.
    
    Args:
        mac_type: MAC protocol type ('CSMA' or 'TDMA')
        primary: Index of primary node (0 to N-1)
        pos0: Node positions array of shape (N, 2)
        N: Number of nodes
        t0: Start time in seconds
        eff: Efficiency dictionary with keys 'CS_base', 'IF_ON', 'PER_jam'
        scn: Scenario parameters
        cs: CSMA parameters
        tdma: TDMA parameters
        cfg: Configuration object
    
    Returns:
        PBFTResult object containing all metrics (latency, energy, throughput, etc.)
    """
    result = PBFTResult()
    quorum = 2 * cfg.QUORUM_F + 1
    phase_delays = np.full(3, np.nan)
    
    # ========================================================================
    # PHASE 1: PRE-PREPARE (Primary -> All Replicas)
    # ========================================================================
    delays_phase1 = []
    
    if mac_type == 'CSMA':
        backoff = csma_backoff_delay(0, cs, eff, N, N)
    else:
        backoff = 0.0
    
    t_start_phase1 = t0 + backoff if mac_type == 'CSMA' else next_tdma_start(t0, primary, tdma)
    
    for rx in range(N):
        if rx == primary:
            continue
        d = link_delay(
            mac_type, primary, rx, t_start_phase1, pos0, cfg.R_BPS, cfg.L_CTRL,
            cfg.C, cfg.PL0_DB, cfg.N_PL, cfg.SIGMA_SF, cfg.PER0,
            cfg.PLOSS_X, eff, cs, tdma
        )
        if np.isfinite(d):
            delays_phase1.append(d)
    
    # Check if quorum reached in Phase 1
    if len(delays_phase1) >= quorum - 1:
        phase_delays[0] = max(delays_phase1)
    else:
        result.success = False
        result.views = 1
        result.phase_delays = phase_delays
        return result
    
    # ========================================================================
    # PHASE 2: PREPARE (All Replicas -> All Replicas)
    # ========================================================================
    t1 = t_start_phase1 + phase_delays[0]
    delays_phase2 = []
    
    for tx in range(N):
        if tx == primary:
            continue
        
        if mac_type == 'CSMA':
            backoff = csma_backoff_delay(0, cs, eff, N - 1, N)
        else:
            backoff = 0.0
        
        t_start_phase2 = t1 + backoff if mac_type == 'CSMA' else next_tdma_start(t1, tx, tdma)
        
        for rx in range(N):
            if rx == tx:
                continue
            d = link_delay(
                mac_type, tx, rx, t_start_phase2, pos0, cfg.R_BPS, cfg.L_CTRL,
                cfg.C, cfg.PL0_DB, cfg.N_PL, cfg.SIGMA_SF, cfg.PER0,
                cfg.PLOSS_X, eff, cs, tdma
            )
            if np.isfinite(d):
                delays_phase2.append(d)
    
    # Check if quorum reached in Phase 2
    needed_phase2 = (N - 1) * (quorum - 1)
    if len(delays_phase2) >= needed_phase2:
        phase_delays[1] = max(delays_phase2)
    else:
        result.success = False
        result.views = 1
        result.phase_delays = phase_delays
        return result
    
    # ========================================================================
    # PHASE 3: COMMIT (All Nodes -> All Nodes)
    # ========================================================================
    t2 = t_start_phase2 + phase_delays[1]
    delays_phase3 = []
    
    for tx in range(N):
        if mac_type == 'CSMA':
            backoff = csma_backoff_delay(0, cs, eff, N, N)
        else:
            backoff = 0.0
        
        t_start_phase3 = t2 + backoff if mac_type == 'CSMA' else next_tdma_start(t2, tx, tdma)
        
        for rx in range(N):
            if rx == tx:
                continue
            d = link_delay(
                mac_type, tx, rx, t_start_phase3, pos0, cfg.R_BPS, cfg.L_CTRL,
                cfg.C, cfg.PL0_DB, cfg.N_PL, cfg.SIGMA_SF, cfg.PER0,
                cfg.PLOSS_X, eff, cs, tdma
            )
            if np.isfinite(d):
                delays_phase3.append(d)
    
    # Check if quorum reached in Phase 3
    needed_phase3 = N * (quorum - 1)
    if len(delays_phase3) >= needed_phase3:
        phase_delays[2] = max(delays_phase3)
    else:
        result.success = False
        result.views = 1
        result.phase_delays = phase_delays
        return result
    
    # ========================================================================
    # SUCCESS: Calculate All Metrics
    # ========================================================================
    result.success = True
    result.views = 1
    result.phase_delays = phase_delays
    
    # Total latency (seconds)
    total_latency_sec = float(np.sum(phase_delays))
    result.latency = total_latency_sec
    result.latency_ms = total_latency_sec * 1000.0  # Convert to milliseconds
    
    # Count transmissions and collisions
    n_trans, n_coll = count_transmissions(phase_delays, N)
    result.n_transmissions = n_trans
    result.n_collisions = n_coll
    
    # Calculate energy consumption (µJ)
    result.energy = calculate_energy(
        mac_type, N, scn, cs, tdma, n_trans, total_latency_sec, cfg
    )
    
    # Estimate PER for throughput calculation
    ctx = extract_context(pos0, N, scn, eff)
    est_per = ctx['estPER']
    
    # Calculate throughput (Mbps)
    result.throughput = calculate_throughput(
        mac_type, N, scn, total_latency_sec, result.success, est_per, cfg
    )
    
    # Calculate margin (time before timeout)
    if np.all(np.isfinite(phase_delays)):
        result.margin = cfg.T_TIMEOUT - total_latency_sec
    
    return result


# ============================================================================
# REWARD CALCULATION WITH PHYSICAL UNITS
# ============================================================================

def base_physical_reward(result: PBFTResult, cfg: Config) -> float:
    """
    Base reward calculation using physical metrics (latency, energy, throughput).
    
    Args:
        result: PBFT result containing metrics
        cfg: Configuration with reward parameters
    
    Returns:
        Reward value based on physical metrics
    """
    if not result.success:
        return cfg.FAILURE_REWARD

    # Extract metrics with safety checks
    latency_ms = result.latency_ms if np.isfinite(result.latency_ms) else cfg.T_TIMEOUT * 1000.0
    energy_uj = result.energy if np.isfinite(result.energy) else 200.0
    throughput_mbps = result.throughput if np.isfinite(result.throughput) else 0.0

    # Normalization references
    deadline_ms = max(1.0, cfg.T_TIMEOUT * 1000.0)
    energy_ref_uj = 100.0  # Reference energy in µJ
    throughput_ref_mbps = max(1.0, cfg.SCALE_THROUGHPUT_MBPS)  # ~10 Mbps scale

    # Normalize values (clamp to reasonable range [0, 2])
    latency_norm = min(2.0, latency_ms / deadline_ms)
    energy_norm = min(2.0, energy_uj / energy_ref_uj)
    throughput_norm = min(2.0, throughput_mbps / throughput_ref_mbps)

    # Weight factors from Config
    lambda_l = cfg.LAMBDA_L  # Latency weight
    lambda_e = cfg.LAMBDA_E  # Energy weight
    lambda_t = cfg.LAMBDA_T  # Throughput weight

    # Reward formula: success baseline - penalties + bonus
    reward = 1.0 - lambda_l * latency_norm - lambda_e * energy_norm + lambda_t * throughput_norm
    
    return reward


def qlearning_reward(result: PBFTResult, cfg: Config) -> float:
    """
    Reward function for Q-learning agent.
    
    Args:
        result: PBFT result
        cfg: Configuration
    
    Returns:
        Reward value
    """
    return base_physical_reward(result, cfg)


def qrdqn_reward(result: PBFTResult, cfg: Config, eff: Dict) -> float:
    """
    Reward function for QR-DQN agent with jamming awareness bonus.
    
    Args:
        result: PBFT result
        cfg: Configuration
        eff: Efficiency dictionary with interference info (must contain 'IF_ON')
    
    Returns:
        Reward value with jamming bonus
    """
    # Base reward from physical metrics
    reward_phys = base_physical_reward(result, cfg)

    # Add jamming bonus if interference is active
    jamming_bonus = 1.0 if eff.get("IF_ON", False) else 0.0
    
    return reward_phys + cfg.THETA_JAM * jamming_bonus


def calculate_reward(
    result: PBFTResult,
    cfg: Config,
    agent_type: str,
    eff: Optional[Dict] = None
) -> float:
    """
    Unified reward calculation function for all agent types.
    
    Args:
        result: PBFT result containing metrics
        cfg: Configuration object with reward parameters
        agent_type: Type of agent ('Q-learning' or 'QR-DQN')
        eff: Efficiency and interference parameters (required for QR-DQN)
    
    Returns:
        Calculated reward value
    
    Raises:
        ValueError: If agent_type is unknown or eff is missing for QR-DQN
    """
    if agent_type == 'Q-learning':
        return qlearning_reward(result, cfg)
    elif agent_type == 'QR-DQN':
        if eff is None:
            raise ValueError("eff parameter required for QR-DQN reward calculation")
        return qrdqn_reward(result, cfg, eff)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Must be 'Q-learning' or 'QR-DQN'")


# ============================================================================
# REPLAY BUFFER FOR QR-DQN
# ============================================================================

class ReplayBuffer:
    """Experience replay buffer for QR-DQN with efficient sampling."""
    
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        gamma: float,
        device: torch.device
    ):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum buffer capacity
            batch_size: Size of sampled batches
            gamma: Discount factor (stored for reference)
            device: PyTorch device for tensors
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.memory)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add experience tuple to buffer.
        
        Args:
            state: Current state (numpy array)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state (numpy array)
            done: Episode termination flag (bool)
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                              torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch and return tensors on the correct device.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as PyTorch tensors
        """
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack numpy arrays efficiently
        states_np = np.stack(states).astype(np.float32)
        next_states_np = np.stack(next_states).astype(np.float32)

        # Convert to tensors on device
        states_tensor = torch.from_numpy(states_np).to(self.device)
        next_states_tensor = torch.from_numpy(next_states_np).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration."""
    
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            alpha: Learning rate (step size)
            gamma: Discount factor for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(2))  # 2 actions: CSMA(0), TDMA(1)
        self.visit_counts = defaultdict(int)
    
    def get_action(self, state: Tuple, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Discretized state tuple
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            Action index: 0 (CSMA) or 1 (TDMA)
        """
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(2)
        return int(np.argmax(self.Q[state]))
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple
    ) -> None:
        """
        Q-learning update rule with physical unit rewards.
        
        Args:
            state: Current state tuple
            action: Action taken (0 or 1)
            reward: Reward received
            next_state: Next state tuple
        """
        self.visit_counts[state] += 1
        
        # Q-learning update: Q(s,a) <- Q(s,a) + α[r + γ*max_a' Q(s',a') - Q(s,a)]
        best_next_value = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.Q[state][action]

    # ============================================================================
# QR-DQN NETWORK
# ============================================================================

class QRDQNNetwork(nn.Module):
    """Quantile Regression DQN network for distributional RL."""
    
    def __init__(
        self,
        state_dim: int = 6,
        n_actions: int = 2,
        n_quantiles: int = 51
    ):
        """
        Initialize QR-DQN network.
        
        Args:
            state_dim: Dimension of state vector
            n_actions: Number of actions (2: CSMA, TDMA)
            n_quantiles: Number of quantiles for distributional RL
        """
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_actions = n_actions

        # Network architecture: state -> hidden layers -> quantile values
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Quantile values of shape (batch_size, n_actions, n_quantiles)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0), self.n_actions, self.n_quantiles)


# ============================================================================
# QR-DQN AGENT
# ============================================================================

class QRDQNAgent:
    """QR-DQN agent with experience replay and target network."""
    
    def __init__(
        self,
        state_dim: int = 6,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        n_quantiles: int = 51,
        buffer_size: int = 50000,
        batch_size: int = 64,
        tau_target: float = 0.005,
        device: torch.device = device
    ):
        """
        Initialize QR-DQN agent.
        
        Args:
            state_dim: Dimension of state vector
            lr: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
            n_quantiles: Number of quantiles for distributional RL
            buffer_size: Maximum size of replay buffer
            batch_size: Mini-batch size for training
            tau_target: Soft update coefficient for target network (0 < tau <= 1)
            device: PyTorch device (CPU or CUDA)
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_quantiles = n_quantiles
        self.device = device
        self.batch_size = batch_size
        self.tau_target = tau_target

        # Initialize main network and target network
        self.net = QRDQNNetwork(state_dim, n_actions=2, n_quantiles=n_quantiles).to(self.device)
        self.target_net = QRDQNNetwork(state_dim, n_actions=2, n_quantiles=n_quantiles).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Mixed precision training (GPU only)
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # Quantile midpoints for QR-DQN loss
        self.tau = torch.linspace(0, 1, n_quantiles + 1, device=self.device)
        self.tau_hat = (self.tau[:-1] + self.tau[1:]) / 2.0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, gamma, self.device)

    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State vector (numpy array of shape (state_dim,))
            explore: Whether to use epsilon-greedy exploration
            
        Returns:
            Action index: 0 (CSMA) or 1 (TDMA)
        """
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(2)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            quantiles = self.net(state_tensor)
            q_values = quantiles.mean(dim=2)  # Average over quantiles
            return int(q_values.argmax(dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False
    ) -> None:
        """
        Store experience and perform gradient update if buffer is ready.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Add experience to replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Wait until buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        batch_size = states.size(0)

        # Compute current quantiles for taken actions
        all_quantiles = self.net(states)
        actions_expanded = actions.view(-1, 1, 1).expand(batch_size, 1, self.n_quantiles)
        current_quantiles = all_quantiles.gather(1, actions_expanded).squeeze(1)

        # Compute target quantiles (Double DQN style)
        with torch.no_grad():
            next_all_quantiles = self.target_net(next_states)
            next_q_values = next_all_quantiles.mean(dim=2)
            next_actions = next_q_values.argmax(dim=1)
            next_actions_expanded = next_actions.view(-1, 1, 1).expand(batch_size, 1, self.n_quantiles)
            next_best_quantiles = next_all_quantiles.gather(1, next_actions_expanded).squeeze(1)

            rewards_expanded = rewards.view(-1, 1)
            dones_expanded = dones.view(-1, 1)
            target_quantiles = rewards_expanded + self.gamma * (1.0 - dones_expanded) * next_best_quantiles

        # Quantile Huber loss
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        huber_loss = torch.where(
            td_errors.abs() <= 1.0,
            0.5 * td_errors ** 2,
            td_errors.abs() - 0.5,
        )

        tau_hat = self.tau_hat.view(1, -1, 1)
        weight = torch.abs(tau_hat - (td_errors.detach() < 0).float())

        loss = (weight * huber_loss).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        self.soft_update_target()

    def soft_update_target(self) -> None:
        """Soft update target network using Polyak averaging."""
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(
                self.tau_target * param.data + (1.0 - self.tau_target) * target_param.data
            )

    def hard_update_target(self) -> None:
        """Hard update target network (full copy)."""
        self.target_net.load_state_dict(self.net.state_dict())


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_qlearning(
    scn: ScenarioParams,
    cs: CSMAParams,
    tdma: TDMAParams,
    cfg: Config,
    n_episodes: int = 100
) -> Tuple[QLearningAgent, List[float]]:
    """
    Train Q-Learning agent with physical unit rewards.
    
    Args:
        scn: Scenario parameters
        cs: CSMA parameters
        tdma: TDMA parameters
        cfg: Configuration object
        n_episodes: Number of training episodes
        
    Returns:
        Tuple of (trained_agent, episode_rewards)
    """
    print(f"\n🎓 Training Q-Learning agent for {n_episodes} episodes...")
    
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.2)
    episode_rewards = []

    for ep in range(n_episodes):
        # Initialize episode
        pos = init_positions(scn.N, scn.area)
        vel = init_velocities(scn.N, scn.v_mean, scn.v_std)
        t = 0.0
        episode_reward = 0.0

        while t < cfg.T_SIM:
            # Create efficiency dictionary
            eff = {
                'CS_base': 0.1,
                'IF_ON': np.random.rand() < scn.if_prob,
                'PER_jam': 0.0
            }
            
            # Extract context and discretize state
            ctx = extract_context(pos, scn.N, scn, eff)
            state = discretize_context(ctx)

            # Select action
            action = agent.get_action(state, explore=True)
            mac_type = 'CSMA' if action == 0 else 'TDMA'

            # Execute PBFT round
            primary = np.random.randint(scn.N)
            result = pbft_round(mac_type, primary, pos, scn.N, t, eff, scn, cs, tdma, cfg)

            # Calculate reward
            reward = calculate_reward(result, cfg, 'Q-learning')
            episode_reward += reward

            # Update position and time
            dt = result.latency if np.isfinite(result.latency) else cfg.DT
            dt = max(dt, cfg.DT)  # Ensure minimum time step
            pos = update_positions(pos, vel, dt, scn.area)
            t += dt

            # Get next state
            ctx_next = extract_context(pos, scn.N, scn, eff)
            next_state = discretize_context(ctx_next)

            # Q-learning update
            agent.update(state, action, reward, next_state)

        episode_rewards.append(episode_reward)

        # Progress logging
        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"  Episode {ep+1}/{n_episodes}, Avg Reward (last 20): {avg_reward:.2f}")

    print(f"✓ Q-Learning training complete!")
    return agent, episode_rewards


def train_qrdqn(
    scn: ScenarioParams,
    cs: CSMAParams,
    tdma: TDMAParams,
    cfg: Config,
    n_episodes: int = 100
) -> Tuple[QRDQNAgent, List[float]]:
    """
    Train QR-DQN agent with physical unit rewards.
    
    Args:
        scn: Scenario parameters
        cs: CSMA parameters
        tdma: TDMA parameters
        cfg: Configuration object
        n_episodes: Number of training episodes
        
    Returns:
        Tuple of (trained_agent, episode_rewards)
    """
    print(f"\n🎓 Training QR-DQN agent for {n_episodes} episodes...")
    
    agent = QRDQNAgent(
        state_dim=6,
        lr=1e-3,
        gamma=0.95,
        epsilon=0.2,
        n_quantiles=51,
        buffer_size=50000,
        batch_size=64,
        tau_target=0.005,
        device=device
    )

    episode_rewards = []

    for ep in range(n_episodes):
        # Initialize episode
        pos = init_positions(scn.N, scn.area)
        vel = init_velocities(scn.N, scn.v_mean, scn.v_std)
        t = 0.0
        episode_reward = 0.0

        while t < cfg.T_SIM:
            # Create efficiency dictionary
            eff = {
                'CS_base': 0.1,
                'IF_ON': np.random.rand() < scn.if_prob,
                'PER_jam': 0.0
            }
            
            # Extract context and convert to state vector
            ctx = extract_context(pos, scn.N, scn, eff)
            state_vec = context_to_state_vector(ctx)

            # Select action via QR-DQN
            action = agent.get_action(state_vec, explore=True)
            mac_type = 'CSMA' if action == 0 else 'TDMA'

            # Execute PBFT round
            primary = np.random.randint(scn.N)
            result = pbft_round(mac_type, primary, pos, scn.N, t, eff, scn, cs, tdma, cfg)

            # Calculate reward with jamming bonus
            reward = calculate_reward(result, cfg, 'QR-DQN', eff)
            episode_reward += reward

            # Update position and time
            dt = result.latency if np.isfinite(result.latency) else cfg.DT
            dt = max(dt, cfg.DT)  # Ensure minimum time step
            next_t = t + dt
            done = next_t >= cfg.T_SIM

            pos = update_positions(pos, vel, dt, scn.area)
            t = next_t

            # Get next state
            ctx_next = extract_context(pos, scn.N, scn, eff)
            next_state_vec = context_to_state_vector(ctx_next)

            # QR-DQN update with replay buffer
            agent.update(state_vec, action, reward, next_state_vec, done=done)

        episode_rewards.append(episode_reward)

        # Progress logging
        if (ep + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"  Episode {ep+1}/{n_episodes}, Avg Reward (last 20): {avg_reward:.2f}")

    print(f"✓ QR-DQN training complete!")
    return agent, episode_rewards


# ============================================================================
# POLICY EVALUATION
# ============================================================================

def evaluate_policy(
    policy_name: str,
    agent: Optional[object],
    scn: ScenarioParams,
    cs: CSMAParams,
    tdma: TDMAParams,
    cfg: Config,
    n_runs: int = 50
) -> ResultsAccumulator:
    """
    Evaluate a policy and collect all metrics.
    
    Args:
        policy_name: Name of policy ('CSMA-only', 'TDMA-only', 'Q-learning', 'QR-DQN')
        agent: Trained agent (None for baseline policies)
        scn: Scenario parameters
        cs: CSMA parameters
        tdma: TDMA parameters
        cfg: Configuration object
        n_runs: Number of evaluation runs
        
    Returns:
        ResultsAccumulator with latency, energy, throughput, etc.
    """
    print(f"  Evaluating {policy_name}...")
    results = ResultsAccumulator()

    for run in range(n_runs):
        # Initialize run
        pos = init_positions(scn.N, scn.area)
        vel = init_velocities(scn.N, scn.v_mean, scn.v_std)
        t = 0.0

        while t < cfg.T_SIM:
            # Create efficiency dictionary
            eff = {
                'CS_base': 0.1,
                'IF_ON': np.random.rand() < scn.if_prob,
                'PER_jam': 0.0
            }
            
            # Extract context
            ctx = extract_context(pos, scn.N, scn, eff)

            # Policy selection
            if policy_name == 'CSMA-only':
                action = 0
            elif policy_name == 'TDMA-only':
                action = 1
            elif policy_name == 'Q-learning':
                state = discretize_context(ctx)
                action = agent.get_action(state, explore=False)
            elif policy_name == 'QR-DQN':
                state_vec = context_to_state_vector(ctx)
                action = agent.get_action(state_vec, explore=False)
            else:
                raise ValueError(f"Unknown policy name: {policy_name}")

            # Execute PBFT round
            mac_type = 'CSMA' if action == 0 else 'TDMA'
            primary = np.random.randint(scn.N)
            result = pbft_round(mac_type, primary, pos, scn.N, t, eff, scn, cs, tdma, cfg)

            # Store all metrics
            results.latency.append(result.latency)
            results.latency_ms.append(result.latency_ms)
            results.energy.append(result.energy)
            results.throughput.append(result.throughput)
            results.n_trans.append(result.n_transmissions)
            results.n_coll.append(result.n_collisions)
            results.success.append(result.success)
            results.views.append(result.views)
            results.mac_choices.append(action)

            # Update position and time
            dt = result.latency if np.isfinite(result.latency) else cfg.DT
            dt = max(dt, cfg.DT)  # Ensure minimum time step
            pos = update_positions(pos, vel, dt, scn.area)
            t += dt

    return results


def evaluate_all_policies(
    scn: ScenarioParams,
    cs: CSMAParams,
    tdma: TDMAParams,
    cfg: Config,
    ql_agent: QLearningAgent,
    qrdqn_agent: QRDQNAgent
) -> Dict[str, ResultsAccumulator]:
    """
    Evaluate all policies: CSMA-only, TDMA-only, Q-learning, QR-DQN.
    
    Args:
        scn: Scenario parameters
        cs: CSMA parameters
        tdma: TDMA parameters
        cfg: Configuration object
        ql_agent: Trained Q-Learning agent
        qrdqn_agent: Trained QR-DQN agent
        
    Returns:
        Dictionary mapping policy names to their ResultsAccumulator
    """
    print(f"\n📊 Evaluating all policies on scenario: {scn.name}")

    all_results = {}

    policies = [
        ('CSMA-only', None),
        ('TDMA-only', None),
        ('Q-learning', ql_agent),
        ('QR-DQN', qrdqn_agent)
    ]

    for policy_name, agent in policies:
        results = evaluate_policy(policy_name, agent, scn, cs, tdma, cfg, n_runs=50)
        all_results[policy_name] = results

    print(f"✓ All policies evaluated!")
    return all_results
        
    self.Q[state][action] += self.alpha * td_error

