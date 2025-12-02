# main.py

#change to  understand if pull works fine  
from pbft_mac.analysis import run_full_pipeline

if __name__ == "__main__":
    all_results, summary_df = run_full_pipeline(output_dir="./pbft_results/")
    print("\n🎉 Analysis complete! Check the output directory for:")
    print("  - Detailed CSV histories")
    print("  - Summary tables (CSV + LaTeX)")
    print("  - Decision analysis crosstabs")
    print("  - Training curves")
    print("  - CDF plots for all metrics")
    print("  - Energy-latency tradeoff plots")
    print("  - Throughput comparisons")
    print("  - Regime maps (heatmaps)")
    print("  - Action distribution plots")
    print("  - Multi-scenario comparison")

print("\n✅ PART 3: Analysis, Visualization & Export - READY")
print("   - CSV export functions implemented")
print("   - Summary table generation (CSV + LaTeX)")
print("   - Decision analysis crosstabs")
print("   - All visualization functions ready")
print("   - Statistics & reporting functions")
print("   - Complete pipeline orchestration")
print("\n🎉 ALL 3 PARTS COMPLETE! Ready to run full simulation.")
print("\nAnalysis complete. Check ./pbft_results/")
