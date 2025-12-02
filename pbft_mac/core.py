# pbft_mac/core.py
#updated conistency  
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from enum import IntEnum  # if you actually use it

# paste: Config, PBFTResult, ResultsAccumulator, ScenarioParams, CSMAParams,
# TDMAParams, per_from_PL, dB_to_per_bump, mobility functions, MAC functions,
# energy/throughput, extract_context


@dataclass
class Config:
    """Enhanced configuration with physical unit scaling constants"""
    
    # Physical unit scaling factors
    SCALE_ENERGY_UJ: float = 100.0        # Scale to ~100 µJ range
    SCALE_LATENCY_MS: float = 50.0        # Scale to ~50 ms range  
    SCALE_THROUGHPUT_MBPS: float = 10.0   # Scale to ~10 Mbps range
    
    # Noise parameters (for realism)
    NOISE_ENERGY_UJ: float = 2.0          # ±2 µJ noise
    NOISE_LATENCY_MS: float = 5.0         # ±5 ms noise
    NOISE_THROUGHPUT_MBPS: float = 0.5    # ±0.5 Mbps noise
    
    # Reward function weights
    W_ENERGY: float = 0.01      # Weight for energy term (µJ)
    W_LATENCY: float = 1.0      # Weight for latency term (ms)
    W_THROUGHPUT: float = 10.0  # Weight for throughput term (Mbps)
    
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
# ENHANCED DATA STRUCTURES
# ============================================================================

@dataclass
class PBFTResult:
    """Enhanced PBFT result with energy and throughput metrics"""
    success: bool = False
    latency: float = np.nan      # Raw latency (seconds)
    latency_ms: float = np.nan   # NEW: Latency in milliseconds
    
    energy: float = np.nan       # NEW: Energy consumption (µJ)
    throughput: float = np.nan   # NEW: Throughput (Mbps)
    
    n_transmissions: int = 0     # NEW: Number of transmissions
    n_collisions: int = 0        # NEW: Estimated collisions
    
    views: int = 0
    phase_delays: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    margin: float = np.nan

@dataclass
class ResultsAccumulator:
    """Enhanced results accumulator with new metrics"""
    latency: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)  # NEW
    
    energy: List[float] = field(default_factory=list)      # NEW
    throughput: List[float] = field(default_factory=list)  # NEW
    
    n_trans: List[int] = field(default_factory=list)       # NEW
    n_coll: List[int] = field(default_factory=list)        # NEW
    
    success: List[bool] = field(default_factory=list)
    views: List[int] = field(default_factory=list)
    mac_choices: List[int] = field(default_factory=list)
    
    ctx_dCenters: List[float] = field(default_factory=list)
    ctx_meanD: List[float] = field(default_factory=list)
    ctx_estPER: List[float] = field(default_factory=list)
    ctx_N: List[int] = field(default_factory=list)
    ctx_if_prob: List[float] = field(default_factory=list)
    ctx_bgLoad: List[float] = field(default_factory=list)

# ============================================================================
# SCENARIO & MAC PARAMETERS (Existing structures)
# ============================================================================

@dataclass
class ScenarioParams:
    """Scenario parameters"""
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
    tx_power_mw: float = 100.0    # NEW: Transmission power (mW)
    idle_power_mw: float = 10.0   # NEW: Idle power (mW)

@dataclass
class TDMAParams:
    """TDMA parameters"""
    Nslot: int = 6
    slotT: float = 0.01
    ctrl_overhead: float = 0.001
    sync_jitter: float = 0.0001
    p_slot_miss: float = 0.05
    tx_power_mw: float = 80.0     # NEW: Transmission power (mW)
    idle_power_mw: float = 5.0    # NEW: Idle power (mW)

@dataclass
class Config:
    """Enhanced configuration with physical unit scaling constants"""

    # ... your existing fields ...

    # Reward weights (shared by Q-learning and QR-DQN)
    LAMBDA_L: float = 0.6   # latency weight
    LAMBDA_E: float = 0.2   # energy weight
    LAMBDA_T: float = 0.5   # throughput weight
    THETA_JAM: float = 0.3  # jamming robustness bonus

    # Failure penalty (for unsuccessful PBFT round)
    FAILURE_REWARD: float = -1.0

# ============================================================================
# ============================================================================
# UTILITY FUNCTIONS (Existing)
# ============================================================================

def per_from_PL(PL: float, PER0: float) -> float:
    """Convert path loss to packet error rate"""
    return min(1.0, PER0 * (1 + 0.01 * max(0, PL - 60)))

def dB_to_per_bump(dB: float) -> float:
    """Convert dB interference to PER increase"""
    return min(0.5, 0.01 * max(0, dB))

# ============================================================================
# KINEMATICS & MOBILITY (Existing)
# ============================================================================

def init_positions(N: int, area: float) -> np.ndarray:
    """Initialize random positions in 2D area"""
    return np.random.rand(N, 2) * area

def init_velocities(N: int, v_mean: float, v_std: float) -> np.ndarray:
    """Initialize random velocities"""
    speeds = np.abs(np.random.randn(N) * v_std + v_mean)
    angles = np.random.rand(N) * 2 * np.pi
    return np.column_stack([speeds * np.cos(angles), speeds * np.sin(angles)])

def update_positions(pos: np.ndarray, vel: np.ndarray, dt: float, area: float) -> np.ndarray:
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
# MAC PROTOCOL FUNCTIONS (From your provided code)
# ============================================================================

def csma_backoff_delay(k_attempt: int, CS: CSMAParams, eff: Dict, K_active: int, N: int) -> float:
    """Calculate CSMA backoff delay"""
    base = max(0, eff['CS_base'])
    jam = 0.15 * float(eff['IF_ON'])
    crowd = CS.kFactor * max(0, (K_active-1)/max(1, N-1))
    pcoll_eff = min(0.95, base + jam + crowd)
    
    # Number of attempts
    nattempt = 1 + np.sum(np.random.rand(CS.retry) < pcoll_eff)
    
    # Binary exponential backoff
    acc = 0.0
    cwi = CS.cwmin
    for a in range(int(nattempt)):
        cwi = min(2*cwi, CS.cwmax)
        bo_slots = max(0, np.random.randint(0, max(1, int(cwi))))
        acc += bo_slots * CS.slot
        if a > 0:
            acc += CS.slot
    
    return acc

def next_tdma_start(t0: float, tx: int, TDMA: TDMAParams) -> float:
    """Calculate next TDMA slot start time"""
    slotT_eff = TDMA.slotT + TDMA.ctrl_overhead / max(1, TDMA.Nslot)
    slotIdx = 1 + (tx % (TDMA.Nslot - 2))
    frameT = TDMA.Nslot * slotT_eff
    
    tInFrm = t0 % frameT
    slotStart = slotIdx * slotT_eff
    
    if tInFrm <= slotStart:
        tstart = t0 + (slotStart - tInFrm)
    else:
        tstart = t0 + (frameT - tInFrm) + slotStart
    
    # Add sync jitter
    if TDMA.sync_jitter > 0:
        tstart += max(0, np.random.randn() * TDMA.sync_jitter)
    
    return tstart

def link_delay(macType: str, tx: int, rx: int, tstart: float, pos0: np.ndarray,
               R_BPS: float, L_CTRL: float, C: float, PL0_dB: float, nPL: float,
               sigmaSF: float, PER0: float, PLOSS_X: float, eff: Dict,
               CS: CSMAParams, TDMA: TDMAParams) -> float:
    """Calculate link delay"""
    ptx = pos0[tx]
    prx = pos0[rx]
    dist = np.linalg.norm(ptx - prx)
    
    # Path loss and PER
    PL = PL0_dB + 10*nPL*np.log10(max(dist, 1)) + sigmaSF*np.random.randn()
    PER = per_from_PL(PL, PER0) + PLOSS_X
    PER = min(1.0, PER + dB_to_per_bump(eff['PER_jam']))
    
    # TDMA slot miss
    if macType == 'TDMA' and np.random.rand() < TDMA.p_slot_miss:
        return np.inf
    
    # Packet drop
    if np.random.rand() < PER:
        return np.inf
    
    Tair = L_CTRL / R_BPS
    Tprop = dist / C
    
    return Tair + Tprop

# ============================================================================
# NEW: ENERGY & THROUGHPUT CALCULATION FUNCTIONS
# ============================================================================

def count_transmissions(phaseDelay: np.ndarray, N: int) -> Tuple[int, int]:
    """
    Count number of transmissions based on PBFT phases
    Returns: (n_transmissions, n_collisions_estimated)
    """
    n_trans = 0
    n_coll = 0
    
    # Pre-prepare: 1 transmission (primary)
    if np.isfinite(phaseDelay[0]):
        n_trans += 1
    
    # Prepare: N-1 transmissions (all except primary)
    if np.isfinite(phaseDelay[1]):
        n_trans += (N - 1)
        n_coll += max(0, int((N - 1) * 0.1))  # ~10% collision rate
    
    # Commit: N transmissions (all nodes)
    if np.isfinite(phaseDelay[2]):
        n_trans += N
        n_coll += max(0, int(N * 0.1))
    
    return n_trans, n_coll

def calculate_energy(macType: str, N: int, SCN: ScenarioParams, CS: CSMAParams, 
                    TDMA: TDMAParams, n_transmissions: int, latency_sec: float,
                    cfg: Config) -> float:
    """
    Calculate energy consumption in microjoules (µJ)
    Energy = transmission_energy + idle_energy + noise
    """
    if not np.isfinite(latency_sec) or latency_sec <= 0:
        return np.nan
    
    # Get power parameters based on MAC type
    if macType == 'CSMA':
        tx_power = CS.tx_power_mw
        idle_power = CS.idle_power_mw
    else:  # TDMA
        tx_power = TDMA.tx_power_mw
        idle_power = TDMA.idle_power_mw
    
    # Estimate transmission time (~1ms per transmission)
    tx_time_sec = n_transmissions * 0.001
    idle_time_sec = max(0, latency_sec - tx_time_sec)
    
    # Energy = Power * Time (mW * s = mJ)
    tx_energy_mj = tx_power * tx_time_sec
    idle_energy_mj = idle_power * idle_time_sec
    total_energy_mj = tx_energy_mj + idle_energy_mj
    
    # Convert mJ to µJ (1 mJ = 1000 µJ)
    energy_uj = total_energy_mj * 1000.0
    
    # Apply scaling and noise
    energy_uj_scaled = cfg.SCALE_ENERGY_UJ * (energy_uj / 100.0)
    energy_uj_scaled += np.random.randn() * cfg.NOISE_ENERGY_UJ
    
    return max(0, energy_uj_scaled)

def calculate_throughput(macType: str, N: int, SCN: ScenarioParams, latency_sec: float,
                        success: bool, estPER: float, cfg: Config) -> float:
    """
    Calculate throughput in Mbps
    Throughput depends on MAC efficiency, PER, and load
    """
    if not success or not np.isfinite(latency_sec) or latency_sec <= 0:
        return 0.0
    
    # MAC efficiency factor
    if macType == 'CSMA':
        alpha = 0.7  # CSMA efficiency
    else:  # TDMA
        alpha = 0.9  # TDMA higher efficiency
    
    # Throughput model: T = α * (1-PER) * (1-0.5*load)
    T_normalized = alpha * (1 - estPER) * (1 - 0.5 * SCN.bgLoad)
    
    # Scale to Mbps with noise
    T_mbps = cfg.SCALE_THROUGHPUT_MBPS * T_normalized
    T_mbps += np.random.randn() * cfg.NOISE_THROUGHPUT_MBPS
    
    return max(0, T_mbps)

# ============================================================================
# CONTEXT EXTRACTION (Existing function - kept as is)
# ============================================================================

def extract_context(pos: np.ndarray, N: int, SCN: ScenarioParams,
                    eff: Dict) -> Dict[str, float]:
    """Extract context features from current state"""
    # Pairwise distances
    dists = []
    for i in range(N):
        for j in range(i+1, N):
            dists.append(np.linalg.norm(pos[i] - pos[j]))
    
    if len(dists) == 0:
        dists = [0.0]
    
    # Estimate PER from mean distance
    meanD = float(np.mean(dists))
    PL_est = 40 + 10*2.5*np.log10(max(meanD, 1))
    estPER = per_from_PL(PL_est, 0.05)
    
    # Context dictionary
    ctx = {
        'dCenters': float(np.std(dists)) if len(dists) > 1 else 0.0,
        'meanD': meanD,
        'estPER': estPER,
        'N': N,
        'if_prob': SCN.if_prob,
        'bgLoad': SCN.bgLoad
    }
    
    return ctx


