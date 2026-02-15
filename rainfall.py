import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
import matplotlib.pyplot as plt


def preprocess_rainfall_sequences(rainfall_scenarios: List[pd.DataFrame],norm_method='global_max') -> Dict:
    print("\n" + "="*70)
    print("PHASE 3: RAINFALL TEMPORAL SEQUENCE PREPROCESSING")
    print("="*70)
    
    num_scenarios = len(rainfall_scenarios)
    
    # -------------------------------------------------------------------------
    # 3.1 EXTRACT RAINFALL SEQUENCES
    # -------------------------------------------------------------------------
    print(f"\n[3.1] Extracting Rainfall Sequences")
    print("-" * 70)
    
    sequences = []
    sequence_lengths = []
    
    for i, scenario_df in enumerate(rainfall_scenarios, 1):
        # Extract intensity column
        if 'intensity_mmhr' in scenario_df.columns:
            intensity = scenario_df['intensity_mmhr'].values
        elif 'intensity' in scenario_df.columns:
            intensity = scenario_df['intensity'].values
        else:
            raise ValueError(f"Scenario {i}: No intensity column found. Columns: {scenario_df.columns.tolist()}")
        
        sequences.append(intensity)
        sequence_lengths.append(len(intensity))
        
        print(f"  RS{i:02d}: {len(intensity)} timesteps, "
              f"peak={intensity.max():.2f} mm/hr at t={intensity.argmax()}")
    
    # Convert to array
    sequences = np.array(sequences)  # Shape: (num_scenarios, num_timesteps)
    
    print(f"\n✓ Extracted {num_scenarios} sequences")
    print(f"  Shape: {sequences.shape}")
    
    # Check all same length
    unique_lengths = set(sequence_lengths)
    if len(unique_lengths) > 1:
        print(f"  ⚠ WARNING: Inconsistent sequence lengths: {unique_lengths}")
    else:
        print(f"  ✓ All sequences have {sequence_lengths[0]} timesteps")
    
    # -------------------------------------------------------------------------
    # 3.2 NORMALIZE RAINFALL SEQUENCES
    # -------------------------------------------------------------------------
    print(f"\n[3.2] Normalizing Sequences (method: {norm_method})")
    print("-" * 70)
    
    if norm_method == 'global_max':
        # Find global maximum across all scenarios and timesteps
        global_max = sequences.max()
        global_min = sequences.min()
        
        print(f"  Global intensity range: [{global_min:.4f}, {global_max:.4f}] mm/hr")
        
        # Normalize
        sequences_norm = sequences / global_max
        
        norm_params = {
            'method': 'global_max',
            'global_max': float(global_max),
            'global_min': float(global_min)
        }
        
        print(f"  Normalization: intensity / {global_max:.4f}")
        print(f"  Normalized range: [{sequences_norm.min():.4f}, {sequences_norm.max():.4f}]")
    
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")
    
    print(f"\n✓ Normalization complete")
    
    # -------------------------------------------------------------------------
    # 3.3 QUALITY CHECKS
    # -------------------------------------------------------------------------
    print(f"\n[3.3] Quality Checks")
    print("-" * 70)
    
    checks = {}
    
    # Shape check
    expected_shape = (num_scenarios, sequence_lengths[0])
    shape_ok = sequences_norm.shape == expected_shape
    print(f"  Shape: {sequences_norm.shape} == {expected_shape} {'✓' if shape_ok else '✗'}")
    checks['shape_ok'] = shape_ok
    
    # No NaNs
    no_nans = not np.isnan(sequences_norm).any()
    print(f"  NaN check: {'✓ None found' if no_nans else '✗ NaNs detected'}")
    checks['no_nans'] = no_nans
    
    # Value range [0, 1]
    in_range = (sequences_norm.min() >= 0) and (sequences_norm.max() <= 1)
    print(f"  Range [0,1]: [{sequences_norm.min():.4f}, {sequences_norm.max():.4f}] {'✓' if in_range else '✗'}")
    checks['in_range'] = in_range
    
    # Temporal pattern preservation (check peak positions unchanged)
    original_peaks = sequences.argmax(axis=1)
    normalized_peaks = sequences_norm.argmax(axis=1)
    peaks_preserved = np.array_equal(original_peaks, normalized_peaks)
    print(f"  Peak timing preserved: {'✓' if peaks_preserved else '✗'}")
    checks['peaks_preserved'] = peaks_preserved
    
    # Statistics summary
    print(f"\n  Summary across {num_scenarios} scenarios:")
    print(f"    Mean peak intensity: {sequences_norm.max(axis=1).mean():.4f}")
    print(f"    Peak timing range: t={original_peaks.min()} to t={original_peaks.max()}")
    print(f"    Mean sequence sum: {sequences_norm.sum(axis=1).mean():.4f}")
    
    checks['all_passed'] = all(checks.values())
    
    results = {
        'sequences': sequences_norm,  # Shape: (num_scenarios, num_timesteps)
        'original_sequences': sequences,  # Keep for reference
        'normalization_params': norm_params,
        'num_scenarios': num_scenarios,
        'num_timesteps': sequences_norm.shape[1],
        'quality_checks': checks,
        'peak_indices': original_peaks.tolist(),
        'peak_values_normalized': sequences_norm.max(axis=1).tolist()
    }
    
    print(f"\n{'='*70}")
    print(f"RAINFALL PREPROCESSING COMPLETE")
    print(f"  Output shape: {sequences_norm.shape}")
    print(f"  All checks passed: {checks['all_passed']}")
    print(f"{'='*70}\n")
    
    return results


def visualize_rainfall_sequences(results: Dict, num_to_plot: int = 20, figsize=(15, 10)):    
    sequences = results['sequences']
    num_scenarios = min(num_to_plot, results['num_scenarios'])
    
    fig, axes = plt.subplots(4, 5, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_scenarios):
        ax = axes[i]
        timesteps = np.arange(results['num_timesteps'])
        
        ax.plot(timesteps, sequences[i], 'b-', linewidth=2, marker='o', markersize=4)
        ax.fill_between(timesteps, sequences[i], alpha=0.3)
        
        # Mark peak
        peak_idx = results['peak_indices'][i]
        peak_val = results['peak_values_normalized'][i]
        ax.plot(peak_idx, peak_val, 'ro', markersize=8, label=f'Peak: {peak_val:.2f}')
        
        ax.set_title(f'RS{i+1}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Timestep', fontsize=8)
        ax.set_ylabel('Normalized Intensity', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    
    plt.tight_layout()
    plt.suptitle('Normalized Rainfall Sequences (All 20 Scenarios)', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    return fig