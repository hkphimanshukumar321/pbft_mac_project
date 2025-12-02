# pbft_mac/analysis.py
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional
import seaborn as sns

from .core import (
    Config,
    ScenarioParams,
    CSMAParams,
    TDMAParams,
    ResultsAccumulator,
)

from .rl import (
    train_qlearning,
    train_qrdqn,
    evaluate_all_policies,
)


"""
PBFT MAC Selection - PART 3: Analysis, Visualization & Export
Complete analysis pipeline with physical units (Energy µJ, Latency ms, Throughput Mbps)
"""


# ============================================================================
# CSV EXPORT FUNCTIONS
# ============================================================================

def export_detailed_history(
    results: ResultsAccumulator,
    scenario_id: str,
    policy_name: str,
    output_dir: str = "./"
) -> str:
    """
    Export detailed round-by-round history to CSV.
    
    Args:
        results: ResultsAccumulator with all metrics
        scenario_id: Scenario identifier/name
        policy_name: Name of the policy
        output_dir: Directory to save CSV files
        
    Returns:
        Filename of exported CSV
    """
    n_rounds = len(results.latency)
    
    data = {
        'scenario_id': [scenario_id] * n_rounds,
        'policy_name': [policy_name] * n_rounds,
        'round': list(range(n_rounds)),
        
        # MAC selection
        'mac_selected': results.mac_choices,
        'mac_name': ['CSMA' if m == 0 else 'TDMA' for m in results.mac_choices],
        
        # Physical unit metrics
        'latency_ms': results.latency_ms,
        'energy_uj': results.energy,
        'throughput_mbps': results.throughput,
        
        # Transmission metrics
        'n_transmissions': results.n_trans,
        'n_collisions': results.n_coll,
        
        # PBFT metrics
        'success': results.success,
        'views': results.views,
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = f"{output_dir}detailed_history_{scenario_id}_{policy_name.replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Exported: {filename}")
    
    return filename


def generate_summary_table(
    all_results: Dict[str, Dict[str, ResultsAccumulator]],
    output_dir: str = "./"
) -> pd.DataFrame:
    """
    Generate summary table across all scenarios and policies.
    
    Args:
        all_results: Nested dict {scenario_name: {policy_name: ResultsAccumulator}}
        output_dir: Directory to save output files
        
    Returns:
        DataFrame with summary statistics
    """
    rows = []
    
    for scenario_name, policies in all_results.items():
        for policy_name, results in policies.items():
            # Filter successful rounds
            success_mask = np.array(results.success, dtype=bool)
            
            if np.sum(success_mask) == 0:
                continue
            
            latency_ms = np.array(results.latency_ms)[success_mask]
            energy_uj = np.array(results.energy)[success_mask]
            throughput_mbps = np.array(results.throughput)[success_mask]
            
            row = {
                'scenario': scenario_name,
                'policy': policy_name,
                'success_rate': np.mean(success_mask) * 100,
                'mean_latency_ms': np.nanmean(latency_ms),
                'std_latency_ms': np.nanstd(latency_ms),
                'p95_latency_ms': np.nanpercentile(latency_ms, 95),
                'mean_energy_uj': np.nanmean(energy_uj),
                'std_energy_uj': np.nanstd(energy_uj),
                'mean_throughput_mbps': np.nanmean(throughput_mbps),
                'std_throughput_mbps': np.nanstd(throughput_mbps),
                'mean_n_trans': np.mean(np.array(results.n_trans)[success_mask]),
                'mean_n_coll': np.mean(np.array(results.n_coll)[success_mask]),
            }
            rows.append(row)
    
    df_summary = pd.DataFrame(rows)
    
    # Save CSV
    csv_file = f"{output_dir}summary_table_all_scenarios.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"✅ Exported: {csv_file}")
    
    # Save LaTeX table
    latex_file = f"{output_dir}summary_table_all_scenarios.tex"
    latex_str = df_summary.to_latex(index=False, float_format="%.2f")
    with open(latex_file, 'w') as f:
        f.write(latex_str)
    print(f"✅ Exported: {latex_file}")
    
    return df_summary


def generate_decision_summary(
    results: ResultsAccumulator,
    scenario_id: str,
    policy_name: str,
    output_dir: str = "./"
) -> pd.DataFrame:
    """
    Generate decision analysis: crosstab of MAC selection vs context.
    Shows when CSMA vs TDMA is chosen under different conditions.
    
    Note: This function requires context features to be stored in ResultsAccumulator.
    Current implementation uses placeholder data as ResultsAccumulator doesn't 
    store context by default.
    
    Args:
        results: ResultsAccumulator with decision history
        scenario_id: Scenario identifier
        policy_name: Name of the policy
        output_dir: Directory to save output files
        
    Returns:
        Crosstab DataFrame of MAC choice vs PER
    """
    n_rounds = len(results.mac_choices)
    
    df = pd.DataFrame({
        'mac_name': ['CSMA' if m == 0 else 'TDMA' for m in results.mac_choices],
        'estPER_bin': ['Unknown'] * n_rounds,
        'meanD_bin': ['Unknown'] * n_rounds,
    })
    
    # Crosstab: MAC choice vs PER
    ct_per = pd.crosstab(df['mac_name'], df['estPER_bin'], normalize='columns') * 100
    
    # Crosstab: MAC choice vs Distance
    ct_dist = pd.crosstab(df['mac_name'], df['meanD_bin'], normalize='columns') * 100
    
    # Save
    ct_per.to_csv(f"{output_dir}decision_vs_PER_{scenario_id}_{policy_name.replace(' ', '_')}.csv")
    ct_dist.to_csv(f"{output_dir}decision_vs_Distance_{scenario_id}_{policy_name.replace(' ', '_')}.csv")
    
    print(f"✅ Exported decision summaries for {policy_name}")
    
    return ct_per


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(
    ql_rewards: List[float],
    qrdqn_rewards: List[float],
    output_dir: str = "./"
):
    """
    Plot training reward curves for both agents.
    
    Args:
        ql_rewards: Q-Learning episode rewards
        qrdqn_rewards: QR-DQN episode rewards
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ql_rewards, label='Q-Learning', alpha=0.7, color='blue')
    ax.plot(qrdqn_rewards, label='QR-DQN', alpha=0.7, color='red')
    
    # Moving average
    window = 10
    if len(ql_rewards) >= window:
        ql_smooth = np.convolve(ql_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(ql_rewards)), ql_smooth, 'b-', 
                linewidth=2, label='Q-Learning (MA)')
    
    if len(qrdqn_rewards) >= window:
        qrdqn_smooth = np.convolve(qrdqn_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(qrdqn_rewards)), qrdqn_smooth, 'r-', 
                linewidth=2, label='QR-DQN (MA)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Curves: Physical Unit Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}training_curves.png")


def plot_cdf_comparison(
    all_results: Dict[str, ResultsAccumulator],
    metric: str = 'latency_ms',
    scenario_name: str = "default",
    output_dir: str = "./"
):
    """
    Plot CDF comparison for a specific metric.
    
    Args:
        all_results: Dict of {policy_name: ResultsAccumulator}
        metric: One of 'latency_ms', 'energy', 'throughput'
        scenario_name: Name of scenario for plot title
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = {
        'latency_ms': 'Latency (ms)',
        'energy': 'Energy (µJ)',
        'throughput': 'Throughput (Mbps)'
    }
    
    colors = {
        'CSMA-only': 'blue',
        'TDMA-only': 'red',
        'Q-Learning': 'green',
        'QR-DQN': 'purple'
    }
    
    for policy_name, results in all_results.items():
        if metric == 'latency_ms':
            data = np.array(results.latency_ms)
        elif metric == 'energy':
            data = np.array(results.energy)
        elif metric == 'throughput':
            data = np.array(results.throughput)
        else:
            continue
        
        # Filter finite values
        data = data[np.isfinite(data)]
        
        if len(data) > 0:
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf, label=policy_name, linewidth=2,
                   color=colors.get(policy_name, 'gray'))
    
    ax.set_xlabel(labels.get(metric, metric))
    ax.set_ylabel('CDF')
    ax.set_title(f'{labels.get(metric, metric)} Distribution - {scenario_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}cdf_{metric}_{scenario_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}cdf_{metric}_{scenario_name}.png")


def plot_energy_latency_tradeoff(
    all_results: Dict[str, ResultsAccumulator],
    scenario_name: str = "default",
    output_dir: str = "./"
):
    """
    Scatter plot: Energy vs Latency tradeoff.
    
    Args:
        all_results: Dict of {policy_name: ResultsAccumulator}
        scenario_name: Name of scenario for plot title
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'CSMA-only': 'blue',
        'TDMA-only': 'red',
        'Q-Learning': 'green',
        'QR-DQN': 'purple'
    }
    
    for policy_name, results in all_results.items():
        energy = np.array(results.energy)
        latency_ms = np.array(results.latency_ms)
        
        # Filter finite values
        mask = np.isfinite(energy) & np.isfinite(latency_ms)
        energy = energy[mask]
        latency_ms = latency_ms[mask]
        
        if len(energy) > 0:
            ax.scatter(latency_ms, energy, alpha=0.5, s=20,
                      color=colors.get(policy_name, 'gray'), label=policy_name)
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Energy (µJ)')
    ax.set_title(f'Energy vs Latency Tradeoff - {scenario_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}energy_latency_tradeoff_{scenario_name}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}energy_latency_tradeoff_{scenario_name}.png")


def plot_throughput_comparison(
    all_results: Dict[str, ResultsAccumulator],
    scenario_name: str = "default",
    output_dir: str = "./"
):
    """
    Box plot: Throughput comparison across policies.
    
    Args:
        all_results: Dict of {policy_name: ResultsAccumulator}
        scenario_name: Name of scenario for plot title
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_list = []
    labels_list = []
    
    for policy_name, results in all_results.items():
        throughput = np.array(results.throughput)
        throughput = throughput[np.isfinite(throughput)]
        
        if len(throughput) > 0:
            data_list.append(throughput)
            labels_list.append(policy_name)
    
    if len(data_list) > 0:
        ax.boxplot(data_list, labels=labels_list)
        ax.set_ylabel('Throughput (Mbps)')
        ax.set_title(f'Throughput Comparison - {scenario_name}')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}throughput_comparison_{scenario_name}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}throughput_comparison_{scenario_name}.png")


def plot_regime_map(
    results: ResultsAccumulator,
    policy_name: str,
    scenario_name: str = "default",
    output_dir: str = "./"
):
    """
    Heatmap showing MAC selection in different regimes.
    X-axis: Mean Distance, Y-axis: Estimated PER
    
    Note: This requires context features to be stored in ResultsAccumulator.
    Current implementation uses placeholder data.
    
    Args:
        results: ResultsAccumulator with decision history
        policy_name: Name of the policy
        scenario_name: Name of scenario for plot title
        output_dir: Directory to save plots
    """
    n_rounds = len(results.mac_choices)
    
    # Create dummy data (should be replaced with actual context data if available)
    df = pd.DataFrame({
        'meanD': np.random.uniform(20, 150, n_rounds),
        'estPER': np.random.uniform(0, 0.5, n_rounds),
        'mac': results.mac_choices
    })
    
    # Bin the data
    df['meanD_bin'] = pd.cut(df['meanD'], bins=10)
    df['estPER_bin'] = pd.cut(df['estPER'], bins=10)
    
    # Pivot table: probability of choosing TDMA
    pivot = df.groupby(['estPER_bin', 'meanD_bin'])['mac'].apply(
        lambda x: np.mean(x == 1) * 100 if len(x) > 0 else 0
    ).unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn',
                cbar_kws={'label': '% TDMA Selected'},
                ax=ax, vmin=0, vmax=100)
    ax.set_xlabel('Mean Distance (binned)')
    ax.set_ylabel('Estimated PER (binned)')
    ax.set_title(f'MAC Selection Regime Map - {policy_name} - {scenario_name}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}regime_map_{policy_name.replace(' ', '_')}_{scenario_name}.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}regime_map_{policy_name}_{scenario_name}.png")


def plot_action_distribution(
    all_results: Dict[str, ResultsAccumulator],
    scenario_name: str = "default",
    output_dir: str = "./"
):
    """
    Bar chart: action distribution for each policy.
    
    Args:
        all_results: Dict of {policy_name: ResultsAccumulator}
        scenario_name: Name of scenario for plot title
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (policy_name, results) in enumerate(all_results.items()):
        if idx >= 4:
            break
        
        mac_counts = pd.Series(results.mac_choices).value_counts()
        mac_labels = ['CSMA', 'TDMA']
        mac_values = [mac_counts.get(0, 0), mac_counts.get(1, 0)]
        
        axes[idx].bar(mac_labels, mac_values, color=['blue', 'red'], alpha=0.7)
        axes[idx].set_title(f'{policy_name}')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'MAC Selection Distribution - {scenario_name}')
    plt.tight_layout()
    plt.savefig(f"{output_dir}action_distribution_{scenario_name}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}action_distribution_{scenario_name}.png")


def plot_multi_scenario_comparison(
    all_results_by_scenario: Dict[str, Dict[str, ResultsAccumulator]],
    output_dir: str = "./"
):
    """
    Multi-panel plot comparing all 3 metrics across all scenarios.
    
    Args:
        all_results_by_scenario: Nested dict {scenario: {policy: ResultsAccumulator}}
        output_dir: Directory to save plots
    """
    scenarios = list(all_results_by_scenario.keys())[:3]  # Max 3 scenarios
    
    if len(scenarios) == 0:
        print("⚠️ No scenarios to plot")
        return
    
    fig, axes = plt.subplots(3, len(scenarios), figsize=(6 * len(scenarios), 15))
    
    # Handle case of single scenario
    if len(scenarios) == 1:
        axes = axes.reshape(-1, 1)
    
    metrics = [
        ('latency_ms', 'Latency (ms)'),
        ('energy', 'Energy (µJ)'),
        ('throughput', 'Throughput (Mbps)')
    ]
    
    colors = {
        'CSMA-only': 'blue',
        'TDMA-only': 'red',
        'Q-Learning': 'green',
        'QR-DQN': 'purple'
    }
    
    for row_idx, (metric_key, metric_label) in enumerate(metrics):
        for col_idx, scenario_name in enumerate(scenarios):
            ax = axes[row_idx, col_idx]
            
            policies = all_results_by_scenario[scenario_name]
            
            for policy_name, results in policies.items():
                if metric_key == 'latency_ms':
                    data = np.array(results.latency_ms)
                elif metric_key == 'energy':
                    data = np.array(results.energy)
                elif metric_key == 'throughput':
                    data = np.array(results.throughput)
                else:
                    continue
                
                data = data[np.isfinite(data)]
                
                if len(data) > 0:
                    sorted_data = np.sort(data)
                    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    ax.plot(sorted_data, cdf, label=policy_name, linewidth=2,
                           color=colors.get(policy_name, 'gray'))
            
            ax.set_xlabel(metric_label)
            ax.set_ylabel('CDF')
            ax.set_title(f'{scenario_name}')
            if col_idx == len(scenarios) - 1:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Scenario Comparison: All Metrics', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(f"{output_dir}multi_scenario_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}multi_scenario_comparison.png")


# ============================================================================
# STATISTICS & REPORTING
# ============================================================================

def print_stats(results: ResultsAccumulator, policy_name: str, scenario_name: str):
    """
    Print detailed statistics for a policy.
    
    Args:
        results: ResultsAccumulator with metrics
        policy_name: Name of the policy
        scenario_name: Name of the scenario
    """
    success_mask = np.array(results.success, dtype=bool)
    success_rate = np.mean(success_mask) * 100
    
    if np.sum(success_mask) == 0:
        print(f"\n❌ {policy_name} ({scenario_name}): No successful rounds")
        return
    
    latency_ms = np.array(results.latency_ms)[success_mask]
    energy_uj = np.array(results.energy)[success_mask]
    throughput_mbps = np.array(results.throughput)[success_mask]
    n_trans = np.array(results.n_trans)[success_mask]
    
    print(f"\n📊 {policy_name} - {scenario_name}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Latency (ms): μ={np.nanmean(latency_ms):.2f}, "
          f"σ={np.nanstd(latency_ms):.2f}, P95={np.nanpercentile(latency_ms, 95):.2f}")
    print(f"  Energy (µJ): μ={np.nanmean(energy_uj):.2f}, σ={np.nanstd(energy_uj):.2f}")
    print(f"  Throughput (Mbps): μ={np.nanmean(throughput_mbps):.2f}, "
          f"σ={np.nanstd(throughput_mbps):.2f}")
    print(f"  Transmissions: μ={np.mean(n_trans):.1f}")
    
    # MAC selection
    if len(results.mac_choices) > 0:
        csma_pct = np.mean(np.array(results.mac_choices) == 0) * 100
        print(f"  MAC Selection: CSMA={csma_pct:.1f}%, TDMA={100-csma_pct:.1f}%")


def export_latex_table(df_summary: pd.DataFrame, output_dir: str = "./"):
    """
    Export publication-ready LaTeX table.
    
    Args:
        df_summary: Summary DataFrame
        output_dir: Directory to save LaTeX file
    """
    # Select key columns
    df_latex = df_summary[['scenario', 'policy', 'success_rate',
                           'mean_latency_ms', 'mean_energy_uj', 'mean_throughput_mbps']]
    
    # Rename columns
    df_latex.columns = ['Scenario', 'Policy', 'Success (%)',
                       'Latency (ms)', 'Energy (µJ)', 'Throughput (Mbps)']
    
    # Generate LaTeX
    latex_str = df_latex.to_latex(
        index=False,
        float_format="%.2f",
        caption="Performance Comparison Across Scenarios and Policies",
        label="tab:performance"
    )
    
    filename = f"{output_dir}performance_table.tex"
    with open(filename, 'w') as f:
        f.write(latex_str)
    
    print(f"✅ Exported LaTeX table: {filename}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(output_dir: str = "./results/"):
    """
    Complete end-to-end pipeline:
    1. Define scenarios
    2. Train agents
    3. Evaluate all policies
    4. Generate all visualizations and exports
    
    Args:
        output_dir: Directory to save all output files
        
    Returns:
        Tuple of (all_results_by_scenario, df_summary)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("🚀 PBFT MAC SELECTION - FULL PIPELINE WITH PHYSICAL UNITS")
    print("="*80)
    
    # Configuration - Use UPPERCASE naming for consistency with other files
    cfg = Config()
    CS = CSMAParams()
    TDMA = TDMAParams()
    
    # Define 3 scenarios
    scenarios = [
        ScenarioParams(N=4, area=100, v_mean=5, v_std=2, if_prob=0.1, 
                      bgLoad=0.3, name="Low-Density"),
        ScenarioParams(N=6, area=100, v_mean=10, v_std=3, if_prob=0.2, 
                      bgLoad=0.5, name="Medium-Density"),
        ScenarioParams(N=8, area=150, v_mean=15, v_std=5, if_prob=0.3, 
                      bgLoad=0.7, name="High-Density"),
    ]
    
    all_results_by_scenario = {}
    
    for SCN in scenarios:
        print(f"\n{'='*80}")
        print(f"📡 SCENARIO: {SCN.name}")
        print(f"{'='*80}")
        
        # Train agents
        print("\n🎓 Training Q-Learning...")
        ql_agent, ql_rewards, env_state = train_qlearning(SCN, CS, TDMA, cfg, n_episodes=100)
        
        print("\n🎓 Training QR-DQN...")
        qrdqn_agent, qrdqn_rewards, env_state = train_qrdqn(SCN, CS, TDMA, cfg, n_episodes=100)
        
        # Plot training curves
        plot_training_curves(ql_rewards, qrdqn_rewards, output_dir)
        
        # Evaluate all policies
        all_results = evaluate_all_policies(SCN, CS, TDMA, cfg, ql_agent, qrdqn_agent)
        all_results_by_scenario[SCN.name] = all_results
        
        # Print statistics
        for policy_name, results in all_results.items():
            print_stats(results, policy_name, SCN.name)
        
        # Export detailed history
        for policy_name, results in all_results.items():
            export_detailed_history(results, SCN.name, policy_name, output_dir)
        
        # Generate decision summaries
        for policy_name, results in all_results.items():
            if policy_name in ['Q-Learning', 'QR-DQN']:
                generate_decision_summary(results, SCN.name, policy_name, output_dir)
        
        # Visualizations for this scenario
        plot_cdf_comparison(all_results, 'latency_ms', SCN.name, output_dir)
        plot_cdf_comparison(all_results, 'energy', SCN.name, output_dir)
        plot_cdf_comparison(all_results, 'throughput', SCN.name, output_dir)
        plot_energy_latency_tradeoff(all_results, SCN.name, output_dir)
        plot_throughput_comparison(all_results, SCN.name, output_dir)
        plot_action_distribution(all_results, SCN.name, output_dir)
        
        # Regime maps for RL policies
        for policy_name, results in all_results.items():
            if policy_name in ['Q-Learning', 'QR-DQN']:
                plot_regime_map(results, policy_name, SCN.name, output_dir)
    
    # Multi-scenario comparison
    print(f"\n{'='*80}")
    print("📊 Generating Multi-Scenario Analysis")
    print(f"{'='*80}")
    
    plot_multi_scenario_comparison(all_results_by_scenario, output_dir)
    
    # Summary table across all scenarios
    df_summary = generate_summary_table(all_results_by_scenario, output_dir)
    export_latex_table(df_summary, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ PIPELINE COMPLETE!")
    print(f"📁 All results saved to: {output_dir}")
    print(f"{'='*80}")
    
    return all_results_by_scenario, df_summary

