# pbft_mac/core.py
# Corrected for consistency with rl.py - all naming standardized

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Enhanced configuration with physical unit scaling constants and reward parameters"""
    
    # Physical unit scaling factors
    SCALE_ENERGY_UJ: float = 100.0        # Scale to ~100 µJ range
    SCALE_LATENCY_MS: float = 50.0        # Scale to ~50 ms range  
    SCALE_THROUGHPUT_MBPS: float = 10.0   # Scale to ~10 Mbps range
    
    # Noise parameters (for realism)
    NOISE_ENERGY_UJ: float = 2.0          # ±2 µJ noise
    NOISE_LATENCY_MS: float = 5.0         # ±5 ms noise
    NOISE_THROUGHPUT_MBPS: float = 0.5    # ±0.5 Mbps noise
    
    # Reward function weights (shared by Q-learning and QR-DQN)
    LAMBDA_L: float = 0.6       # Latency weight
    LAMBDA_E: float = 0.2       # Energy weight
    LAMBDA_T: float = 0.5       # Throughput weight
    THETA_JAM: float = 0.3      # Jamming robustness bonus
    
    # Failure penalty (for unsuccessful PBFT round)
    FAILURE_REWARD: float = -1.0
    
    # Simulation parameters
    T_SIM: float = 30.0         # Simulation time (seconds)
    DT: float = 0.1             # Time step (seconds)
    
    # PBFT parameters
    QUORUM_F: int = 1           # Fault tolerance (N >= 3f+1)
    T_TIMEOUT: float = 1.0      # Timeout (seconds)
    
    # Radio parameters
    R_BPS: float = 1e6          # Data rate (1 Mbps)
    L_CTRL: float = 256.0       # Control message size (bytes)
    C: float = 3e8              # Speed of light (m/s)
    PL0_DB: float = 40.0        # Path loss at 1m (dB)
    N_PL: float = 2.5           # Path loss exponent
    SIGMA_SF: float = 4.0       # Shadow fading std dev (dB)
    PER0: float = 0.05          # Base packet error rate
    PLOSS_X: float = 0.02       # Additional loss factor


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PBFTResult:
    """Enhanced PBFT result with energy and throughput metrics"""
    success: bool = False
    latency: float = np.nan      # Raw latency (seconds)
    latency_ms: float = np.nan   # Latency in milliseconds
    
    energy: float = np.nan       # Energy consumption (µJ)
    throughput: float = np.nan   # Throughput (Mbps)
    
    n_transmissions: int = 0     # Number of transmissions
    n_collisions: int = 0        # Estimated collisions
    
    views: int = 0
    phase_delays: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    margin: float = np.nan


@dataclass
class ResultsAccumulator:
    """Enhanced results accumulator with all metrics"""
    latency: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)
    
    energy: List[float] = field(default_factory=list)
    throughput: List[float] = field(default_factory=list)
    
    n_trans: List[int] = field(default_factory=list)
    n_coll: List[int] = field(default_factory=list)
    
    success: List[bool] = field(default_factory=list)
    views: List[int] = field(default_factory=list)
    mac_choices: List[int] = field(default_factory=list)
    
    ctx_dCenters: List[float] = field(default_factory=list)
    ctx_meanD: List[float] = field(default_factory=list)
    ctx_estPER: List[float] = field(default_factory=list)
    ctx_N: List[int] = field(default_factory=list)
    ctx_if_prob: List[float] = field(default_factory=list)
    ctx_bgLoad: List[float] = field(default_factory=list)


@dataclass
class ScenarioParams:
    """Scenario parameters for network simulation"""
    N: int = 4
    area: float = 100.0
    v_mean: float = 5.0
    v_std: float = 2.0
    if_prob: float = 0.1
    bgLoad: float = 0.3
    name: str = "default"


@dataclass
class CSMAParams:
    """CSMA/CA parameters"""
    cwmin: float = 8.0
    cwmax: float = 128.0
    slot: float = 0.00001
    retry: int = 5
    kFactor: float = 0.3
    tx_power_mw: float = 100.0    # Transmission power (mW)
    idle_power_mw: float = 10.0   # Idle power (mW)


@dataclass
class TDMAParams:
    """TDMA parameters"""
    Nslot: int = 6
    slotT: float = 0.01
    ctrl_overhead: float = 0.001
    sync_jitter: float = 0.0001
    p_slot_miss: float = 0.05
    tx_power_mw: float = 80.0     # Transmission power (mW)
    idle_power_mw: float = 5.0    # Idle power (mW)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def per_from_PL(path_loss: float, per0: float) -> float:
    """
    Convert path loss to packet error rate.
    
    Args:
        path_loss: Path loss in dB
        per0: Base packet error rate
    
    Returns:
        Packet error rate (clamped to [0, 1])
    """
    return min(1.0, per0 * (1 + 0.01 * max(0, path_loss - 60)))


def dB_to_per_bump(dB: float) -> float:
    """
    Convert dB interference to PER increase.
    
    Args:
        dB: Interference level in dB
    
    Returns:
        PER increase (clamped to [0, 0.5])
    """
    return min(0.5, 0.01 * max(0, dB))


# ============================================================================
# KINEMATICS & MOBILITY
# ============================================================================

def init_positions(N: int, area: float) -> np.ndarray:
    """
    Initialize random positions in 2D area.
    
    Args:
        N: Number of nodes
        area: Side length of square area
    
    Returns:
        Array of shape (N, 2) with positions
    """
    return np.random.rand(N, 2) * area


def init_velocities(N: int, v_mean: float, v_std: float) -> np.ndarray:
    """
    Initialize random velocities for nodes.
    
    Args:
        N: Number of nodes
        v_mean: Mean speed
        v_std: Standard deviation of speed
    
    Returns:
        Array of shape (N, 2) with velocity vectors
    """
    speeds = np.abs(np.random.randn(N) * v_std + v_mean)
    angles = np.random.rand(N) * 2 * np.pi
    return np.column_stack([speeds * np.cos(angles), speeds * np.sin(angles)])


def update_positions(pos: np.ndarray, vel: np.ndarray, dt: float, area: float) -> np.ndarray:
    """
    Update positions with reflective boundary conditions.
    
    Args:
        pos: Current positions (N, 2)
        vel: Velocities (N, 2) - modified in-place for reflections
        dt: Time step
        area: Side length of square area
    
    Returns:
        Updated positions (N, 2)
    """
    pos_new = pos + vel * dt

    # Reflect on lower boundary (0)
    lower_mask = pos_new < 0.0
    pos_new[lower_mask] = -pos_new[lower_mask]
    vel[lower_mask] = -vel[lower_mask]

    # Reflect on upper boundary (area)
    upper_mask = pos_new > area
    pos_new[upper_mask] = 2.0 * area - pos_new[upper_mask]
    vel[upper_mask] = -vel[upper_mask]

    return pos_new


# ============================================================================
# MAC PROTOCOL FUNCTIONS
# ============================================================================

def csma_backoff_delay(
    k_attempt: int,
    cs: CSMAParams,
    eff: Dict,
    k_active: int,
    N: int
) -> float:
    """
    Calculate CSMA backoff delay with collision probability.
    
    Args:
        k_attempt: Current attempt number
        cs: CSMA parameters
        eff: Efficiency dictionary with 'CS_base' and 'IF_ON' keys
        k_active: Number of active nodes
        N: Total number of nodes
    
    Returns:
        Backoff delay in seconds
    """
    base = max(0, eff['CS_base'])
    jam = 0.15 * float(eff['IF_ON'])
    crowd = cs.kFactor * max(0, (k_active - 1) / max(1, N - 1))
    pcoll_eff = min(0.95, base + jam + crowd)
    
    # Number of collision attempts
    n_attempt = 1 + np.sum(np.random.rand(cs.retry) < pcoll_eff)
    
    # Binary exponential backoff
    acc = 0.0
    cwi = cs.cwmin
    for a in range(int(n_attempt)):
        cwi = min(2 * cwi, cs.cwmax)
        bo_slots = max(0, np.random.randint(0, max(1, int(cwi))))
        acc += bo_slots * cs.slot
        if a > 0:
            acc += cs.slot
    
    return acc


def next_tdma_start(t0: float, tx: int, tdma: TDMAParams) -> float:
    """
    Calculate next TDMA slot start time for a given node.
    
    Args:
        t0: Current time
        tx: Transmitting node index
        tdma: TDMA parameters
    
    Returns:
        Start time of next TDMA slot for node tx
    """
    slot_t_eff = tdma.slotT + tdma.ctrl_overhead / max(1, tdma.Nslot)
    slot_idx = 1 + (tx % (tdma.Nslot - 2))
    frame_t = tdma.Nslot * slot_t_eff
    
    t_in_frame = t0 % frame_t
    slot_start = slot_idx * slot_t_eff
    
    if t_in_frame <= slot_start:
        t_start = t0 + (slot_start - t_in_frame)
    else:
        t_start = t0 + (frame_t - t_in_frame) + slot_start
    
    # Add sync jitter
    if tdma.sync_jitter > 0:
        t_start += max(0, np.random.randn() * tdma.sync_jitter)
    
    return t_start


def link_delay(
    mac_type: str,
    tx: int,
    rx: int,
    t_start: float,
    pos0: np.ndarray,
    r_bps: float,
    l_ctrl: float,
    c: float,
    pl0_db: float,
    n_pl: float,
    sigma_sf: float,
    per0: float,
    ploss_x: float,
    eff: Dict,
    cs: CSMAParams,
    tdma: TDMAParams
) -> float:
    """
    Calculate link delay between transmitter and receiver.
    
    Args:
        mac_type: MAC protocol type ('CSMA' or 'TDMA')
        tx: Transmitter node index
        rx: Receiver node index
        t_start: Start time of transmission
        pos0: Node positions array (N, 2)
        r_bps: Data rate in bits per second
        l_ctrl: Control message size in bytes
        c: Speed of light (m/s)
        pl0_db: Path loss at 1m (dB)
        n_pl: Path loss exponent
        sigma_sf: Shadow fading standard deviation (dB)
        per0: Base packet error rate
        ploss_x: Additional loss factor
        eff: Efficiency dictionary with 'PER_jam' key
        cs: CSMA parameters
        tdma: TDMA parameters
    
    Returns:
        Link delay in seconds (np.inf if packet dropped)
    """
    ptx = pos0[tx]
    prx = pos0[rx]
    dist = np.linalg.norm(ptx - prx)
    
    # Path loss and PER calculation
    path_loss = pl0_db + 10 * n_pl * np.log10(max(dist, 1)) + sigma_sf * np.random.randn()
    per = per_from_PL(path_loss, per0) + ploss_x
    per = min(1.0, per + dB_to_per_bump(eff['PER_jam']))
    
    # TDMA slot miss
    if mac_type == 'TDMA' and np.random.rand() < tdma.p_slot_miss:
        return np.inf
    
    # Packet drop due to channel error
    if np.random.rand() < per:
        return np.inf
    
    # Air time and propagation delay
    t_air = l_ctrl / r_bps
    t_prop = dist / c
    
    return t_air + t_prop


# ============================================================================
# ENERGY & THROUGHPUT CALCULATION
# ============================================================================

def count_transmissions(phase_delays: np.ndarray, N: int) -> Tuple[int, int]:
    """
    Count number of transmissions and estimate collisions based on PBFT phases.
    
    Args:
        phase_delays: Array of 3 phase delays [pre-prepare, prepare, commit]
        N: Number of nodes
    
    Returns:
        Tuple of (n_transmissions, n_collisions_estimated)
    """
    n_trans = 0
    n_coll = 0
    
    # Phase 1 (Pre-prepare): 1 transmission from primary
    if np.isfinite(phase_delays[0]):
        n_trans += 1
    
    # Phase 2 (Prepare): N-1 transmissions (all except primary)
    if np.isfinite(phase_delays[1]):
        n_trans += (N - 1)
        n_coll += max(0, int((N - 1) * 0.1))  # Estimate ~10% collision rate
    
    # Phase 3 (Commit): N transmissions (all nodes)
    if np.isfinite(phase_delays[2]):
        n_trans += N
        n_coll += max(0, int(N * 0.1))  # Estimate ~10% collision rate
    
    return n_trans, n_coll


def calculate_energy(
    mac_type: str,
    N: int,
    scn: ScenarioParams,
    cs: CSMAParams,
    tdma: TDMAParams,
    n_transmissions: int,
    latency_sec: float,
    cfg: Config
) -> float:
    """
    Calculate energy consumption in microjoules (µJ).
    Energy = transmission_energy + idle_energy + noise
    
    Args:
        mac_type: MAC protocol type ('CSMA' or 'TDMA')
        N: Number of nodes
        scn: Scenario parameters
        cs: CSMA parameters
        tdma: TDMA parameters
        n_transmissions: Number of transmissions
        latency_sec: Total latency in seconds
        cfg: Configuration object
    
    Returns:
        Energy consumption in µJ (np.nan if invalid)
    """
    if not np.isfinite(latency_sec) or latency_sec <= 0:
        return np.nan
    
    # Get power parameters based on MAC type
    if mac_type == 'CSMA':
        tx_power_mw = cs.tx_power_mw
        idle_power_mw = cs.idle_power_mw
    else:  # TDMA
        tx_power_mw = tdma.tx_power_mw
        idle_power_mw = tdma.idle_power_mw
    
    # Estimate transmission time (~1ms per transmission)
    tx_time_sec = n_transmissions * 0.001
    idle_time_sec = max(0, latency_sec - tx_time_sec)
    
    # Energy = Power * Time (mW * s = mJ)
    tx_energy_mj = tx_power_mw * tx_time_sec
    idle_energy_mj = idle_power_mw * idle_time_sec
    total_energy_mj = tx_energy_mj + idle_energy_mj
    
    # Convert mJ to µJ (1 mJ = 1000 µJ)
    energy_uj = total_energy_mj * 1000.0
    
    # Apply scaling and noise
    energy_uj_scaled = cfg.SCALE_ENERGY_UJ * (energy_uj / 100.0)
    energy_uj_scaled += np.random.randn() * cfg.NOISE_ENERGY_UJ
    
    return max(0, energy_uj_scaled)


def calculate_throughput(
    mac_type: str,
    N: int,
    scn: ScenarioParams,
    latency_sec: float,
    success: bool,
    est_per: float,
    cfg: Config
) -> float:
    """
    Calculate throughput in Mbps.
    Throughput depends on MAC efficiency, PER, and background load.
    
    Args:
        mac_type: MAC protocol type ('CSMA' or 'TDMA')
        N: Number of nodes
        scn: Scenario parameters
        latency_sec: Total latency in seconds
        success: Whether PBFT round succeeded
        est_per: Estimated packet error rate
        cfg: Configuration object
    
    Returns:
        Throughput in Mbps (0.0 if unsuccessful or invalid)
    """
    if not success or not np.isfinite(latency_sec) or latency_sec <= 0:
        return 0.0
    
    # MAC efficiency factor
    if mac_type == 'CSMA':
        alpha = 0.7  # CSMA efficiency
    else:  # TDMA
        alpha = 0.9  # TDMA higher efficiency
    
    # Throughput model: T = α * (1-PER) * (1-0.5*load)
    t_normalized = alpha * (1 - est_per) * (1 - 0.5 * scn.bgLoad)
    
    # Scale to Mbps with noise
    t_mbps = cfg.SCALE_THROUGHPUT_MBPS * t_normalized
    t_mbps += np.random.randn() * cfg.NOISE_THROUGHPUT_MBPS
    
    return max(0, t_mbps)


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

def extract_context(
    pos: np.ndarray,
    N: int,
    scn: ScenarioParams,
    eff: Dict
) -> Dict[str, float]:
    """
    Extract context features from current network state.
    
    Args:
        pos: Node positions array (N, 2)
        N: Number of nodes
        scn: Scenario parameters
        eff: Efficiency dictionary
    
    Returns:
        Dictionary with context features:
        - dCenters: Standard deviation of pairwise distances
        - meanD: Mean pairwise distance
        - estPER: Estimated packet error rate
        - N: Number of nodes
        - if_prob: Interference probability
        - bgLoad: Background load
    """
    # Calculate all pairwise distances
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dists.append(np.linalg.norm(pos[i] - pos[j]))
    
    if len(dists) == 0:
        dists = [0.0]
    
    # Estimate PER from mean distance
    mean_d = float(np.mean(dists))
    pl_est = 40 + 10 * 2.5 * np.log10(max(mean_d, 1))
    est_per = per_from_PL(pl_est, 0.05)
    
    # Construct context dictionary
    ctx = {
        'dCenters': float(np.std(dists)) if len(dists) > 1 else 0.0,
        'meanD': mean_d,
        'estPER': est_per,
        'N': N,
        'if_prob': scn.if_prob,
        'bgLoad': scn.bgLoad
    }
    
    return ctx
