#!/usr/bin/env python3
"""
LLM Throughput Benchmark Analysis
Read benchmark results and perform data analysis and visualization
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score
import seaborn as sns

# Configure fonts (include Chinese support)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)

def load_data(filename='throughput_benchmark.xlsx'):
    """Load Excel data"""
    try:
        df = pd.read_excel(filename)
        print(f"✅ Successfully loaded data from {filename}")
        print(f"   Data shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filename} not found!")
        print("   Please run graph.py first to generate benchmark data.")
        return None

def analyze_basic_statistics(df):
    """Basic statistical analysis"""
    print("\n" + "="*70)
    print("BASIC STATISTICS")
    print("="*70)
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\n--- Key Insights ---")
    
    # Total Throughput analysis
    max_total_idx = df['Total Throughput (tokens/s)'].idxmax()
    print(f"\n📊 Total Throughput:")
    print(f"   Max: {df.loc[max_total_idx, 'Total Throughput (tokens/s)']:.2f} tokens/s at {df.loc[max_total_idx, 'Workers']} workers")
    print(f"   Min: {df['Total Throughput (tokens/s)'].min():.2f} tokens/s")
    print(f"   Mean: {df['Total Throughput (tokens/s)'].mean():.2f} tokens/s")
    print(f"   Std: {df['Total Throughput (tokens/s)'].std():.2f} tokens/s")
    
    # Avg Throughput analysis
    max_avg_idx = df['Avg Throughput (tokens/s)'].idxmax()
    print(f"\n📊 Average Throughput per Worker:")
    print(f"   Max: {df.loc[max_avg_idx, 'Avg Throughput (tokens/s)']:.2f} tokens/s at {df.loc[max_avg_idx, 'Workers']} workers")
    print(f"   Min: {df['Avg Throughput (tokens/s)'].min():.2f} tokens/s")
    print(f"   Mean: {df['Avg Throughput (tokens/s)'].mean():.2f} tokens/s")
    print(f"   Std: {df['Avg Throughput (tokens/s)'].std():.2f} tokens/s")
    
    # First Token Time analysis
    min_ftt_idx = df['Avg First Token Time (ms)'].idxmin()
    print(f"\n📊 Average First Token Time:")
    print(f"   Min: {df.loc[min_ftt_idx, 'Avg First Token Time (ms)']:.1f} ms at {df.loc[min_ftt_idx, 'Workers']} workers")
    print(f"   Max: {df['Avg First Token Time (ms)'].max():.1f} ms")
    print(f"   Mean: {df['Avg First Token Time (ms)'].mean():.1f} ms")
    print(f"   Std: {df['Avg First Token Time (ms)'].std():.1f} ms")
    
    # Compute efficiency metric
    df['Efficiency'] = df['Total Throughput (tokens/s)'] / df['Workers']
    print(f"\n📊 Scaling Efficiency (Total Throughput / Workers):")
    print(f"   Best: {df['Efficiency'].max():.2f} at {df.loc[df['Efficiency'].idxmax(), 'Workers']} workers")
    print(f"   Worst: {df['Efficiency'].min():.2f} at {df.loc[df['Efficiency'].idxmin(), 'Workers']} workers")
    
    # Compute gain/loss ratio
    total_gain = df['Total Throughput (tokens/s)'].diff()
    avg_loss = -df['Avg Throughput (tokens/s)'].diff()
    gain_loss_ratio = total_gain / avg_loss
    
    print(f"\n📊 Scaling Gain/Loss Ratio (per additional worker):")
    print(f"   Total Throughput Gain / Avg Throughput Loss")
    valid_ratio = gain_loss_ratio[1:]  # Skip the first NaN
    
    # 检查是否有有效数据
    if valid_ratio.notna().any() and (valid_ratio != 0).any():
        print(f"   Best ratio: {valid_ratio.max():.2f} at {df.loc[valid_ratio.idxmax(), 'Workers']} workers")
        print(f"   Worst ratio: {valid_ratio.min():.2f} at {df.loc[valid_ratio.idxmin(), 'Workers']} workers")
        print(f"   Mean ratio: {valid_ratio.mean():.2f}")
        print(f"   Interpretation: Ratio > 1 means positive scaling, < 1 means diminishing returns")
    else:
        print(f"   ⚠️  No valid data - all throughput values are 0")
        print(f"   Please run graph.py first to generate benchmark data!")
    
    return df

def fit_models(df):
    """Fit spline models"""
    print("\n" + "="*70)
    print("REGRESSION ANALYSIS")
    print("="*70)
    
    X = df['Workers'].values
    
    results = {}
    
    # Spline smoothing parameter
    spline_smoothing = 100.0  # smoothing factor, larger -> smoother
    
    # 1. Total Throughput vs Workers
    y_total = df['Total Throughput (tokens/s)'].values
    
    # UnivariateSpline (smoothing spline)
    spline_total = UnivariateSpline(X, y_total, s=spline_smoothing)
    y_pred_spline = spline_total(X)
    r2_spline = r2_score(y_total, y_pred_spline)
    
    print(f"\n📈 Total Throughput vs Workers:")
    print(f"   Spline R²: {r2_spline:.4f}")
    print(f"   Spline smoothing factor: {spline_smoothing}")
    
    results['total'] = {
        'spline': (spline_total, r2_spline),
        'X': X,
        'y': y_total,
        'y_pred_spline': y_pred_spline
    }
    
    # 2. Avg Throughput vs Workers
    y_avg = df['Avg Throughput (tokens/s)'].values
    
    # UnivariateSpline
    spline_avg = UnivariateSpline(X, y_avg, s=spline_smoothing)
    y_avg_pred_spline = spline_avg(X)
    r2_avg_spline = r2_score(y_avg, y_avg_pred_spline)
    
    print(f"\n📈 Avg Throughput vs Workers:")
    print(f"   Spline R²: {r2_avg_spline:.4f}")
    
    results['avg'] = {
        'spline': (spline_avg, r2_avg_spline),
        'X': X,
        'y': y_avg,
        'y_pred_spline': y_avg_pred_spline
    }
    
    # 3. First Token Time vs Workers
    y_ftt = df['Avg First Token Time (ms)'].values

    # UnivariateSpline - use larger smoothing factor to avoid overfitting
    # First Token Time has a larger dynamic range, so increase s accordingly
    spline_ftt = UnivariateSpline(X, y_ftt, s=spline_smoothing * 50)  # increase by 50x
    y_ftt_pred_spline = spline_ftt(X)
    r2_ftt_spline = r2_score(y_ftt, y_ftt_pred_spline)
    
    print(f"\n📈 First Token Time vs Workers:")
    print(f"   Spline R²: {r2_ftt_spline:.4f}")
    print(f"   Spline smoothing factor: {spline_smoothing * 50}")
    
    results['ftt'] = {
        'spline': (spline_ftt, r2_ftt_spline),
        'X': X,
        'y': y_ftt,
        'y_pred_spline': y_ftt_pred_spline
    }
    
    return results

def create_visualizations(df, regression_results):
    """Create visualizations"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Total Throughput vs Workers (with regression)
    ax1 = plt.subplot(2, 3, 1)
    # Connect all points
    ax1.plot(df['Workers'], df['Total Throughput (tokens/s)'], 'o-', alpha=0.6, linewidth=1.5, markersize=5, label='Actual Data', color='steelblue')
    # Spline fit
    ax1.plot(regression_results['total']['X'], regression_results['total']['y_pred_spline'], 
             'r-', label=f"Spline (R²={regression_results['total']['spline'][1]:.3f})", linewidth=2)
    
    # Annotate feature points
    max_idx = df['Total Throughput (tokens/s)'].idxmax()
    min_idx = df['Total Throughput (tokens/s)'].idxmin()
    ax1.scatter(df.loc[max_idx, 'Workers'], df.loc[max_idx, 'Total Throughput (tokens/s)'], 
                color='red', s=100, zorder=5, marker='*')
    ax1.annotate(f"Max: ({df.loc[max_idx, 'Workers']}, {df.loc[max_idx, 'Total Throughput (tokens/s)']:.1f})",
                xy=(df.loc[max_idx, 'Workers'], df.loc[max_idx, 'Total Throughput (tokens/s)']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=8, ha='left', arrowprops=dict(arrowstyle='->', color='red'))
    ax1.scatter(df.loc[min_idx, 'Workers'], df.loc[min_idx, 'Total Throughput (tokens/s)'], 
                color='blue', s=100, zorder=5, marker='*')
    ax1.annotate(f"Min: ({df.loc[min_idx, 'Workers']}, {df.loc[min_idx, 'Total Throughput (tokens/s)']:.1f})",
                xy=(df.loc[min_idx, 'Workers'], df.loc[min_idx, 'Total Throughput (tokens/s)']),
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                fontsize=8, ha='left', arrowprops=dict(arrowstyle='->', color='blue'))
    
    ax1.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Throughput (tokens/s)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Throughput vs Workers', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Avg Throughput vs Workers (with regression)
    ax2 = plt.subplot(2, 3, 2)
    # Connect all points
    ax2.plot(df['Workers'], df['Avg Throughput (tokens/s)'], 'o-', alpha=0.6, linewidth=1.5, markersize=5, color='orange', label='Actual Data')
    # Spline fit
    ax2.plot(regression_results['avg']['X'], regression_results['avg']['y_pred_spline'], 
             'r-', label=f"Spline (R²={regression_results['avg']['spline'][1]:.3f})", linewidth=2)
    
    # Annotate feature points
    max_idx = df['Avg Throughput (tokens/s)'].idxmax()
    min_idx = df['Avg Throughput (tokens/s)'].idxmin()
    ax2.scatter(df.loc[max_idx, 'Workers'], df.loc[max_idx, 'Avg Throughput (tokens/s)'], 
                color='red', s=100, zorder=5, marker='*')
    ax2.annotate(f"Max: ({df.loc[max_idx, 'Workers']}, {df.loc[max_idx, 'Avg Throughput (tokens/s)']:.1f})",
                xy=(df.loc[max_idx, 'Workers'], df.loc[max_idx, 'Avg Throughput (tokens/s)']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=8, ha='left', arrowprops=dict(arrowstyle='->', color='red'))
    ax2.scatter(df.loc[min_idx, 'Workers'], df.loc[min_idx, 'Avg Throughput (tokens/s)'], 
                color='blue', s=100, zorder=5, marker='*')
    ax2.annotate(f"Min: ({df.loc[min_idx, 'Workers']}, {df.loc[min_idx, 'Avg Throughput (tokens/s)']:.1f})",
                xy=(df.loc[min_idx, 'Workers'], df.loc[min_idx, 'Avg Throughput (tokens/s)']),
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                fontsize=8, ha='left', arrowprops=dict(arrowstyle='->', color='blue'))
    
    ax2.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Avg Throughput (tokens/s)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Throughput per Worker', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. First Token Time vs Workers (with regression)
    ax3 = plt.subplot(2, 3, 3)
    # Connect all points
    ax3.plot(df['Workers'], df['Avg First Token Time (ms)'], 'o-', alpha=0.6, linewidth=1.5, markersize=5, color='green', label='Actual Data')
    # Spline fit
    ax3.plot(regression_results['ftt']['X'], regression_results['ftt']['y_pred_spline'], 
             'r-', label=f"Spline (R²={regression_results['ftt']['spline'][1]:.3f})", linewidth=2)
    
    # 添加特征点标注
    max_idx = df['Avg First Token Time (ms)'].idxmax()
    min_idx = df['Avg First Token Time (ms)'].idxmin()
    ax3.scatter(df.loc[max_idx, 'Workers'], df.loc[max_idx, 'Avg First Token Time (ms)'], 
                color='red', s=100, zorder=5, marker='*')
    ax3.annotate(f"Max: ({df.loc[max_idx, 'Workers']}, {df.loc[max_idx, 'Avg First Token Time (ms)']:.1f})",
                xy=(df.loc[max_idx, 'Workers'], df.loc[max_idx, 'Avg First Token Time (ms)']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                fontsize=8, ha='left', arrowprops=dict(arrowstyle='->', color='red'))
    ax3.scatter(df.loc[min_idx, 'Workers'], df.loc[min_idx, 'Avg First Token Time (ms)'], 
                color='blue', s=100, zorder=5, marker='*')
    ax3.annotate(f"Min: ({df.loc[min_idx, 'Workers']}, {df.loc[min_idx, 'Avg First Token Time (ms)']:.1f})",
                xy=(df.loc[min_idx, 'Workers'], df.loc[min_idx, 'Avg First Token Time (ms)']),
                xytext=(10, -15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                fontsize=8, ha='left', arrowprops=dict(arrowstyle='->', color='blue'))
    
    ax3.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
    ax3.set_ylabel('First Token Time (ms)', fontsize=11, fontweight='bold')
    ax3.set_title('Avg First Token Time vs Workers', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scaling Efficiency - Gain vs Loss Ratio
    ax4 = plt.subplot(2, 3, 4)
    
    # Compute total throughput gain per additional worker
    total_throughput_gain = df['Total Throughput (tokens/s)'].diff()
    
    # Compute avg throughput loss per worker (decrease)
    avg_throughput_loss = -df['Avg Throughput (tokens/s)'].diff()
    
    # Compute gain/loss ratio
    efficiency_ratio = total_throughput_gain / avg_throughput_loss
    
    # Skip the first NaN
    valid_workers = df['Workers'][1:]
    valid_ratio = efficiency_ratio[1:]
    
    ax4.plot(valid_workers, valid_ratio, marker='o', linewidth=2, markersize=6, color='purple', label='Gain/Loss Ratio')
    ax4.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Break-even (ratio=1)')
    ax4.fill_between(valid_workers, 0, valid_ratio, where=(valid_ratio > 1), 
                      alpha=0.3, color='green', label='Positive scaling')
    ax4.fill_between(valid_workers, 0, valid_ratio, where=(valid_ratio <= 1), 
                      alpha=0.3, color='red', label='Negative scaling')
    
    ax4.set_xlabel('Number of Workers', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Gain/Loss Ratio', fontsize=11, fontweight='bold')
    ax4.set_title('Scaling Efficiency: Total Throughput Gain vs Avg Throughput Loss', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # Annotate best and worst points
    if len(valid_ratio) > 0:
        best_idx = valid_ratio.idxmax()
        worst_idx = valid_ratio.idxmin()
        ax4.annotate(f'Best: {valid_ratio[best_idx]:.2f}\n@ {df.loc[best_idx, "Workers"]} workers',
                    xy=(df.loc[best_idx, 'Workers'], valid_ratio[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                    fontsize=8, ha='left')
        ax4.annotate(f'Worst: {valid_ratio[worst_idx]:.2f}\n@ {df.loc[worst_idx, "Workers"]} workers',
                    xy=(df.loc[worst_idx, 'Workers'], valid_ratio[worst_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7),
                    fontsize=8, ha='left')
    
    # 5. Correlation Heatmap
    ax5 = plt.subplot(2, 3, 5)
    corr_data = df[['Workers', 'Total Throughput (tokens/s)', 
                     'Avg Throughput (tokens/s)', 'Avg First Token Time (ms)']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, ax=ax5, cbar_kws={'shrink': 0.8})
    ax5.set_title('Correlation Matrix', fontsize=13, fontweight='bold')
    
    # 6. Box plots for distribution analysis
    ax6 = plt.subplot(2, 3, 6)
    
    # Normalize data for comparison
    normalized_data = df[['Total Throughput (tokens/s)', 
                          'Avg Throughput (tokens/s)', 
                          'Avg First Token Time (ms)']].copy()
    for col in normalized_data.columns:
        normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                               (normalized_data[col].max() - normalized_data[col].min())
    
    box_data = [normalized_data['Total Throughput (tokens/s)'],
                normalized_data['Avg Throughput (tokens/s)'],
                normalized_data['Avg First Token Time (ms)']]
    
    bp = ax6.boxplot(box_data, labels=['Total\nThroughput', 'Avg\nThroughput', 'First Token\nTime'],
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    
        # Add statistical annotations
    for i, (data, color) in enumerate(zip(box_data, ['lightblue', 'lightcoral', 'lightgreen']), 1):
        median_val = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        
        # Annotate median
        ax6.text(i, median_val, f'{median_val:.2f}', 
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkred')
        # Annotate Q1
        ax6.text(i + 0.3, q1, f'Q1:{q1:.2f}', 
            ha='left', va='center', fontsize=7, color='darkblue')
        # Annotate Q3
        ax6.text(i + 0.3, q3, f'Q3:{q3:.2f}', 
            ha='left', va='center', fontsize=7, color='darkblue')
    
    ax6.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
    ax6.set_title('Distribution of Normalized Metrics', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'benchmark_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to {output_file}")
    
    # Show plot
    plt.show()
    
    print("\n📊 Created 6 visualizations:")
    print("   1. Total Throughput vs Workers (connected line + Spline)")
    print("   2. Average Throughput per Worker (connected line + Spline)")
    print("   3. First Token Time vs Workers (connected line + Spline)")
    print("   4. Scaling Efficiency: Gain/Loss Ratio per additional worker")
    print("   5. Correlation Matrix")
    print("   6. Distribution Analysis (Box Plots)")

def main():
    """Main function"""
    print("="*70)
    print("LLM THROUGHPUT BENCHMARK ANALYSIS")
    print("="*70)
    
    # 加载数据
    df = load_data('throughput_benchmark.xlsx')
    if df is None:
        return
    
    # 检查数据是否有效
    if df['Total Throughput (tokens/s)'].sum() == 0:
        print("\n" + "="*70)
        print("❌ ERROR: All throughput values are 0!")
        print("="*70)
        print("\nThis means graph.py failed to collect valid benchmark data.")
        print("Please check:")
        print("  1. Is TensorRT server running on http://localhost:8355?")
        print("  2. Run: python graph.py")
        print("  3. Check for any error messages during benchmark")
        print("\nThen run this analysis script again.")
        print("="*70)
        return
    
    # 基础统计分析
    df = analyze_basic_statistics(df)
    
    # 回归分析
    regression_results = fit_models(df)
    
    # 创建可视化
    create_visualizations(df, regression_results)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
