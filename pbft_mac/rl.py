import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Tuple, List
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

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[RL] Using device: {device}")


# ============================================================================#
# PBFT SIMULATION CORE (Enhanced with Energy & Throughput)
# ============================================================================#

def pbft_round(macType: str, primary: int, pos0: np.ndarray, N: int, t0: float,
               eff: Dict, SCN: ScenarioParams, CS: CSMAParams, TDMA: TDMAParams,
               cfg: Config) -> PBFTResult:
    result = PBFTResult()
    quorum = 2 * cfg.QUORUM_F + 1
    phaseDelay = np.full(3, np.nan)
    
    # Phase 1: Pre-Prepare
    delays = []
    if macType == 'CSMA':
        backoff = csma_backoff_delay(0, CS, eff, N, N)
    else:
        backoff = 0.0
    
    tstart = t0 + backoff if macType == 'CSMA' else next_tdma_start(t0, primary, TDMA)
    
    for rx in range(N):
        if rx == primary:
            continue
        d = link_delay(macType, primary, rx, tstart, pos0, cfg.R_BPS, cfg.L_CTRL,
                      cfg.C, cfg.PL0_DB, cfg.N_PL, cfg.SIGMA_SF, cfg.PER0,
                      cfg.PLOSS_X, eff, CS, TDMA)
        if np.isfinite(d):
            delays.append(d)
    
    if len(delays) >= quorum - 1:
        phaseDelay[0] = max(delays)
    else:
        result.success = False
        result.views = 1
        result.phase_delays = phaseDelay
        return result
    
    # Phase 2: Prepare
    t1 = tstart + phaseDelay[0]
    delays2 = []
    
    for tx in range(N):
        if tx == primary:
            continue
        if macType == 'CSMA':
            backoff = csma_backoff_delay(0, CS, eff, N-1, N)
        else:
            backoff = 0.0
        
        tstart2 = t1 + backoff if macType == 'CSMA' else next_tdma_start(t1, tx, TDMA)
        
        for rx in range(N):
            if rx == tx:
                continue
            d = link_delay(macType, tx, rx, tstart2, pos0, cfg.R_BPS, cfg.L_CTRL,
                          cfg.C, cfg.PL0_DB, cfg.N_PL, cfg.SIGMA_SF, cfg.PER0,
                          cfg.PLOSS_X, eff, CS, TDMA)
            if np.isfinite(d):
                delays2.append(d)
    
    needed = (N - 1) * (quorum - 1)
    if len(delays2) >= needed:
        phaseDelay[1] = max(delays2)
    else:
        result.success = False
        result.views = 1
        result.phase_delays = phaseDelay
        return result
    
    # Phase 3: Commit
    t2 = tstart2 + phaseDelay[1]
    delays3 = []
    
    for tx in range(N):
        if macType == 'CSMA':
            backoff = csma_backoff_delay(0, CS, eff, N, N)
        else:
            backoff = 0.0
        
        tstart3 = t2 + backoff if macType == 'CSMA' else next_tdma_start(t2, tx, TDMA)
        
        for rx in range(N):
            if rx == tx:
                continue
            d = link_delay(macType, tx, rx, tstart3, pos0, cfg.R_BPS, cfg.L_CTRL,
                          cfg.C, cfg.PL0_DB, cfg.N_PL, cfg.SIGMA_SF, cfg.PER0,
                          cfg.PLOSS_X, eff, CS, TDMA)
            if np.isfinite(d):
                delays3.append(d)
    
    needed = N * (quorum - 1)
    if len(delays3) >= needed:
        phaseDelay[2] = max(delays3)
    else:
        result.success = False
        result.views = 1
        result.phase_delays = phaseDelay
        return result
    
    # Success: Calculate all metrics
    result.success = True
    result.views = 1
    result.phase_delays = phaseDelay
    
    # Total latency (seconds)
    total_latency_sec = float(np.sum(phaseDelay))
    result.latency = total_latency_sec
    result.latency_ms = total_latency_sec * 1000.0  # Convert to ms
    
    # Count transmissions and collisions
    n_trans, n_coll = count_transmissions(phaseDelay, N)
    result.n_transmissions = n_trans
    result.n_collisions = n_coll
    
    # Calculate energy (µJ)
    result.energy = calculate_energy(macType, N, SCN, CS, TDMA, n_trans, 
                                     total_latency_sec, cfg)
    
    # Estimate PER for throughput calculation
    ctx = extract_context(pos0, N, SCN, eff)
    estPER = ctx['estPER']
    
    # Calculate throughput (Mbps)
    result.throughput = calculate_throughput(macType, N, SCN, total_latency_sec,
                                            result.success, estPER, cfg)
    
    # Margin calculation
    if np.all(np.isfinite(phaseDelay)):
        result.margin = cfg.T_TIMEOUT - total_latency_sec
    
    return result


# ============================================================================#
# REWARD CALCULATION WITH PHYSICAL UNITS
# ============================================================================#

def base_physical_reward(result: PBFTResult, cfg: Config) -> float:
    """Shared physical reward for both Q-learning and QR-DQN."""
    if not result.success:
        return cfg.FAILURE_REWARD

    # Extract metrics with safe defaults
    L_ms = result.latency_ms if np.isfinite(result.latency_ms) else cfg.T_TIMEOUT * 1000.0
    E_uj = result.energy if np.isfinite(result.energy) else 200.0
    T_mbps = result.throughput if np.isfinite(result.throughput) else 0.0

    # Normalization references
    deadline_ms = max(1.0, cfg.T_TIMEOUT * 1000.0)
    E_ref = 100.0  # reference energy in µJ
    T_ref = max(1.0, cfg.SCALE_THROUGHPUT_MBPS)  # ~10 Mbps scale

    # Normalized metrics (clamped)
    L_norm = min(2.0, L_ms / deadline_ms)
    E_norm = min(2.0, E_uj / E_ref)
    T_norm = min(2.0, T_mbps / T_ref)

    # Reward weights from Config
    lambda_L = cfg.LAMBDA_L
    lambda_E = cfg.LAMBDA_E
    lambda_T = cfg.LAMBDA_T

    s = 1.0  # success branch
    r_phys = s - lambda_L * L_norm - lambda_E * E_norm + lambda_T * T_norm
    return r_phys

def qlearning_reward(result: PBFTResult, cfg: Config) -> float:
    """Q-learning reward – uses shared physical reward."""
    return base_physical_reward(result, cfg)

def qrdqn_reward(result: PBFTResult, cfg: Config, eff: Dict) -> float:
    """QR-DQN reward = shared physical reward + small jamming bonus."""
    r_phys = base_physical_reward(result, cfg)
    delta_jam = 1.0 if eff.get("IF_ON", False) else 0.0
    theta_jam = cfg.THETA_JAM
    return r_phys + theta_jam * delta_jam


# ============================================================================#
# REPLAY BUFFER FOR QR-DQN
# ============================================================================#

class ReplayBuffer:
    """Experience replay buffer for QR-DQN."""
    def __init__(self, buffer_size: int, batch_size: int, gamma: float, device: torch.device):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

    def __len__(self):
        return len(self.memory)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Sample a mini-batch and return tensors on the correct device."""
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack numpy arrays efficiently
        states_np = np.stack(states).astype(np.float32)        # (B, state_dim)
        next_states_np = np.stack(next_states).astype(np.float32)

        states = torch.from_numpy(states_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones


# ============================================================================#
# Q-LEARNING AGENT (Enhanced with Physical Unit Rewards)
# ============================================================================#

class QLearningAgent:
    """Tabular Q-Learning agent with physical unit rewards."""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(2))  # 2 actions: CSMA(0), TDMA(1)
        self.visit_counts = defaultdict(int)
    
    def get_action(self, state: Tuple, explore: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(2)
        return int(np.argmax(self.Q[state]))
    
    def update(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """Q-learning update with physical unit reward."""
        self.visit_counts[state] += 1
        best_next = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state][action]
        
        self.Q[state][action] += self.alpha * td_error


# ============================================================================#
# QR-DQN NETWORK (Enhanced for Physical Unit Rewards)
# ============================================================================#

class QRDQNNetwork(nn.Module):
    """Quantile Regression DQN network."""
    def __init__(self, state_dim: int = 6, n_actions: int = 2, n_quantiles: int = 51):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_actions = n_actions

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0), self.n_actions, self.n_quantiles)


class QRDQNAgent:
    """QR-DQN agent with physical unit rewards."""
    def __init__(self, state_dim: int = 6, lr: float = 1e-3, gamma: float = 0.95,
                 epsilon: float = 0.2, n_quantiles: int = 51, buffer_size: int = 50_000,
                 batch_size: int = 64, device_: torch.device = device):
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_quantiles = n_quantiles
        self.device = device_
        self.batch_size = batch_size

        self.net = QRDQNNetwork(state_dim, n_actions=2, n_quantiles=n_quantiles).to(self.device)
        self.target_net = QRDQNNetwork(state_dim, n_actions=2, n_quantiles=n_quantiles).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        self.tau = torch.linspace(0, 1, n_quantiles + 1, device=self.device)
        self.tau_hat = (self.tau[:-1] + self.tau[1:]) / 2.0

        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, gamma, self.device)

    def get_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Action selection."""
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(2)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            quantiles = self.net(state_t)
            q_values = quantiles.mean(dim=2)
            return int(q_values.argmax(dim=1).item())

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool = False):
        """Replay buffer update and gradient step."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        batch_size = states.size(0)

        all_quantiles = self.net(states)
        actions_expanded = actions.view(-1, 1, 1).expand(batch_size, 1, self.n_quantiles)
        current_quantiles = all_quantiles.gather(1, actions_expanded).squeeze(1)

        with torch.no_grad():
            next_all_quantiles = self.target_net(next_states)
            next_q_values = next_all_quantiles.mean(dim=2)
            next_actions = next_q_values.argmax(dim=1)
            next_actions_expanded = next_actions.view(-1, 1, 1).expand(batch_size, 1, self.n_quantiles)
            next_best_quantiles = next_all_quantiles.gather(1, next_actions_expanded).squeeze(1)

            rewards_expanded = rewards.view(-1, 1)
            dones_expanded = dones.view(-1, 1)
            target_quantiles = rewards_expanded + self.gamma * (1.0 - dones_expanded) * next_best_quantiles

        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)

        huber_loss = torch.where(
            td_errors.abs() <= 1.0,
            0.5 * td_errors ** 2,
            td_errors.abs() - 0.5,
        )

        tau_hat = self.tau_hat.view(1, -1, 1)
        weight = torch.abs(tau_hat - (td_errors.detach() < 0).float())

        loss = (weight * huber_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update_target()

    def soft_update_target(self):
        """Soft update target network."""
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(
                self.tau_target * param.data + (1.0 - self.tau_target) * target_param.data
            )

    def update_target(self):
        """Force a full copy of the target network."""
        self.target_net.load_state_dict(self.net.state_dict())

  
# ============================================================================#
# TRAINING FUNCTIONS (Q-Learning & QR-DQN)
# ============================================================================#

def train_qlearning(SCN: ScenarioParams, CS: CSMAParams, TDMA: TDMAParams,
                   cfg: Config, n_episodes: int = 100) -> Tuple[QLearningAgent, List[float]]:
    """
    Train Q-Learning agent with physical unit rewards
    Returns: (trained_agent, reward_history)
    """
    agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon=0.2)
    reward_history = []

    for ep in range(n_episodes):
        # Initialize episode
        pos = init_positions(SCN.N, SCN.area)
        vel = init_velocities(SCN.N, SCN.v_mean, SCN.v_std)
        t = 0.0
        episode_reward = 0.0

        while t < cfg.T_SIM:
            # Extract context
            eff = {'CS_base': 0.1, 'IF_ON': np.random.rand() < SCN.if_prob, 'PER_jam': 0.0}
            ctx = extract_context(pos, SCN.N, SCN, eff)
            state = discretize_context(ctx)

            # Select action
            action = agent.get_action(state, explore=True)
            macType = 'CSMA' if action == 0 else 'TDMA'

            # Execute PBFT round
            primary = np.random.randint(SCN.N)
            result = pbft_round(macType, primary, pos, SCN.N, t, eff, SCN, CS, TDMA, cfg)

            # Calculate reward with physical units
            reward = qlearning_reward(result, cfg)
            episode_reward += reward

            # Update position
            dt = result.latency if np.isfinite(result.latency) else cfg.DT
            pos = update_positions(pos, vel, dt, SCN.area)
            t += dt

            # Next state
            ctx_next = extract_context(pos, SCN.N, SCN, eff)
            next_state = discretize_context(ctx_next)

            # Q-learning update
            agent.update(state, action, reward, next_state)

        reward_history.append(episode_reward)

        if (ep + 1) % 20 == 0:
            print(f"  Q-Learning Episode {ep+1}/{n_episodes}, Avg Reward: {np.mean(reward_history[-20:]):.2f}")

    return agent, reward_history

    #training  

def train_qrdqn(SCN: ScenarioParams, CS: CSMAParams, TDMA: TDMAParams,
                cfg: Config, n_episodes: int = 100) -> Tuple[QRDQNAgent, List[float]]:
    """
    Train QR-DQN agent with physical unit rewards
    Returns: (trained_agent, reward_history)
    """
    agent = QRDQNAgent(
        state_dim=6, lr=1e-3, gamma=0.95, epsilon=0.2, n_quantiles=51,
        buffer_size=50_000, batch_size=64, device_=device)

    reward_history = []

    for ep in range(n_episodes):
        # Initialize episode
        pos = init_positions(SCN.N, SCN.area)
        vel = init_velocities(SCN.N, SCN.v_mean, SCN.v_std)
        t = 0.0
        episode_reward = 0.0

        while t < cfg.T_SIM:
            # Extract context
            eff = {'CS_base': 0.1, 'IF_ON': np.random.rand() < SCN.if_prob, 'PER_jam': 0.0}
            ctx = extract_context(pos, SCN.N, SCN, eff)

            # State vector (continuous)
            state_vec = np.array([ctx['dCenters'], ctx['meanD'], ctx['estPER'],
                                 ctx['N'], ctx['if_prob'], ctx['bgLoad']], dtype=np.float32)

            # Select action via QR-DQN
            action = agent.get_action(state_vec, explore=True)
            macType = 'CSMA' if action == 0 else 'TDMA'

            # Execute PBFT round
            primary = np.random.randint(SCN.N)
            result = pbft_round(macType, primary, pos, SCN.N, t, eff, SCN, CS, TDMA, cfg)

            # Reward (physical units)
            reward = qrdqn_reward(result, cfg, eff)
            episode_reward += reward

            # Time step for mobility update
            dt = result.latency if np.isfinite(result.latency) else cfg.DT
            dt = max(dt, cfg.DT)  # avoid 0
            next_t = t + dt
            done = next_t >= cfg.T_SIM

            # Update positions
            pos = update_positions(pos, vel, dt, SCN.area)
            t = next_t

            # Next state
            eff_next = eff  # same interference parameters within this small step
            ctx_next = extract_context(pos, SCN.N, SCN, eff_next)
            next_state_vec = np.array([ctx_next['dCenters'], ctx_next['meanD'], ctx_next['estPER'],
                                      ctx_next['N'], ctx_next['if_prob'], ctx_next['bgLoad']], dtype=np.float32)

            # QR-DQN update with replay
            agent.update(state_vec, action, reward, next_state_vec, done=done)

        reward_history.append(episode_reward)

        if (ep + 1) % 20 == 0:
            print(f"  QR-DQN Episode {ep+1}/{n_episodes}, "
                  f"Avg Reward (last 20): {np.mean(reward_history[-20:]):.2f}")

    return agent, reward_history

# ============================================================================#
# POLICY EVALUATION (Evaluation after Training)
# ============================================================================#

def evaluate_policy(policy_name: str, agent, SCN: ScenarioParams, CS: CSMAParams,
                   TDMA: TDMAParams, cfg: Config, n_runs: int = 50) -> ResultsAccumulator:
    """
    Evaluate a policy and collect all metrics.
    Returns: ResultsAccumulator with latency, energy, throughput, etc.
    """
    results = ResultsAccumulator()

    for run in range(n_runs):
        pos = init_positions(SCN.N, SCN.area)
        vel = init_velocities(SCN.N, SCN.v_mean, SCN.v_std)
        t = 0.0

        while t < cfg.T_SIM:
            eff = {'CS_base': 0.1, 'IF_ON': np.random.rand() < SCN.if_prob, 'PER_jam': 0.0}
            ctx = extract_context(pos, SCN.N, SCN, eff)

            # Policy selection
            if policy_name == 'CSMA-only':
                action = 0
            elif policy_name == 'TDMA-only':
                action = 1
            elif policy_name == 'Q-Learning':
                state = discretize_context(ctx)
                action = agent.get_action(state, explore=False)
            elif policy_name == 'QR-DQN':
                state_vec = np.array([ctx['dCenters'], ctx['meanD'], ctx['estPER'],
                                     ctx['N'], ctx['if_prob'], ctx['bgLoad']], dtype=np.float32)
                action = agent.get_action(state_vec, explore=False)
            else:
                action = 0

            macType = 'CSMA' if action == 0 else 'TDMA'
            primary = np.random.randint(SCN.N)
            result = pbft_round(macType, primary, pos, SCN.N, t, eff, SCN, CS, TDMA, cfg)

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

            # Update
            dt = result.latency if np.isfinite(result.latency) else cfg.DT
            pos = update_positions(pos, vel, dt, SCN.area)
            t += dt

    return results

def evaluate_all_policies(SCN: ScenarioParams, CS: CSMAParams, TDMA: TDMAParams,
                         cfg: Config, ql_agent, qrdqn_agent) -> Dict[str, ResultsAccumulator]:
    """
    Evaluate all policies: CSMA-only, TDMA-only, Q-Learning, QR-DQN
    Returns: Dictionary of results for each policy
    """
    print(f"\n📊 Evaluating all policies on scenario: {SCN.name}")

    all_results = {}

    policies = [
        ('CSMA-only', None),
        ('TDMA-only', None),
        ('Q-Learning', ql_agent),
        ('QR-DQN', qrdqn_agent)
    ]

    for policy_name, agent in policies:
        print(f"  Evaluating {policy_name}...")
        results = evaluate_policy(policy_name, agent, SCN, CS, TDMA, cfg, n_runs=50)
        all_results[policy_name] = results

    return all_results






