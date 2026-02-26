import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_scenario_metadata(info_csv_path: str) -> pd.DataFrame:
    """
    Load scenario metadata from info.csv
    
    Expected columns:
    - No: Scenario number (1-50)
    - Pattern Type: SCS-style front-loaded, balanced, back-loaded, or Triangular
    - Total storm depth P (mm): Rainfall depth
    - Peak time fraction r (tpeak / D): Peak timing (only for triangular)
    """
    print("\n[Loading Scenario Metadata]")
    print("-" * 70)
    
    df = pd.read_csv(info_csv_path)
    
    print(f"  Loaded {len(df)} scenarios from {info_csv_path}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Verify required columns
    required = ['No', 'Pattern Type', 'Total storm depth P (mm)']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check scenario range
    if df['No'].min() != 1 or df['No'].max() != 50:
        print(f"  ⚠ WARNING: Scenario numbers range from {df['No'].min()} to {df['No'].max()}")
    
    return df


def create_conditioning_vectors(metadata_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """
    Create conditioning vectors from metadata.
    
    Conditioning vector structure (4 dimensions):
    [pattern_type_encoded, depth_normalized, tpeak, has_tpeak]
    
    Pattern encoding:
    - 0: SCS-style front-loaded
    - 1: SCS-style balanced
    - 2: SCS-style back-loaded
    - 3: Triangular / Chicago-style
    
    Returns:
    --------
    conditioning_vectors : np.ndarray, shape (50, 4)
    encoding_info : Dict with encoding details
    """
    print("\n[Creating Conditioning Vectors]")
    print("-" * 70)
    
    # Pattern type encoding
    pattern_encoding = {
        'SCS‑style front‑loaded (convective – similar to NRCS Type II)': 0,
        'SCS‑style front-loaded (convective – similar to NRCS Type II)': 0,  # Handle variations
        'SCS‑style balanced (intermediate)': 1,
        'SCS-style balanced (intermediate)': 1,
        'SCS‑style back‑loaded (frontal – similar to Type I)': 2,
        'SCS-style back-loaded (frontal – similar to Type I)': 2,
        'Triangular / Chicago‑style with peak at fraction of duration': 3,
        'Triangular / Chicago-style with peak at fraction of duration': 3,
    }
    
    # Find max depth for normalization
    max_depth = metadata_df['Total storm depth P (mm)'].max()
    print(f"  Depth normalization: max = {max_depth:.2f} mm")
    
    conditioning_vectors = []
    
    for idx, row in metadata_df.iterrows():
        scenario_id = row['No']
        pattern_str = row['Pattern Type']
        depth_mm = row['Total storm depth P (mm)']
        
        # Encode pattern type
        if pattern_str not in pattern_encoding:
            # Try to match partial string
            matched = False
            for key in pattern_encoding:
                if 'front' in pattern_str.lower() and 'front' in key.lower():
                    pattern_encoded = pattern_encoding[key]
                    matched = True
                    break
                elif 'balanced' in pattern_str.lower() and 'balanced' in key.lower():
                    pattern_encoded = pattern_encoding[key]
                    matched = True
                    break
                elif 'back' in pattern_str.lower() and 'back' in key.lower():
                    pattern_encoded = pattern_encoding[key]
                    matched = True
                    break
                elif 'triangular' in pattern_str.lower() or 'chicago' in pattern_str.lower():
                    pattern_encoded = pattern_encoding[key]
                    matched = True
                    break
            
            if not matched:
                raise ValueError(f"Unknown pattern type for scenario {scenario_id}: {pattern_str}")
        else:
            pattern_encoded = pattern_encoding[pattern_str]
        
        # Normalize depth
        depth_normalized = depth_mm / max_depth
        
        # Handle tpeak (only for triangular)
        tpeak_col = 'Peak time fraction r (tpeak / D)'
        if tpeak_col in metadata_df.columns and pd.notna(row[tpeak_col]):
            tpeak = float(row[tpeak_col])
            has_tpeak = 1.0
        else:
            tpeak = 0.0
            has_tpeak = 0.0
        
        # Create conditioning vector
        cond_vector = np.array([
            pattern_encoded,
            depth_normalized,
            tpeak,
            has_tpeak
        ], dtype=np.float32)
        
        conditioning_vectors.append(cond_vector)
    
    conditioning_vectors = np.array(conditioning_vectors)  # Shape: (50, 4)
    
    # Summary
    print(f"\n  Conditioning vectors created: {conditioning_vectors.shape}")
    print(f"\n  Pattern type distribution:")
    unique_patterns = np.unique(conditioning_vectors[:, 0])
    pattern_names = ['Front-loaded', 'Balanced', 'Back-loaded', 'Triangular']
    for p in unique_patterns:
        count = np.sum(conditioning_vectors[:, 0] == p)
        print(f"    {pattern_names[int(p)]}: {count} scenarios")
    
    print(f"\n  Depth range: [{conditioning_vectors[:, 1].min():.3f}, "
          f"{conditioning_vectors[:, 1].max():.3f}] (normalized)")
    print(f"  Triangular scenarios with tpeak: {int(conditioning_vectors[:, 3].sum())}")
    
    encoding_info = {
        'pattern_encoding': {v: k for k, v in pattern_encoding.items()},
        'max_depth_mm': float(max_depth),
        'conditioning_dims': 4,
        'dim_names': ['pattern_type', 'depth_normalized', 'tpeak', 'has_tpeak'],
        'pattern_names': pattern_names
    }
    
    return conditioning_vectors, encoding_info


def preprocess_rainfall_sequences(
    rainfall_scenarios: List[pd.DataFrame],
    norm_method='global_max',
    info_csv_path: str = None  # NEW: Path to info.csv
) -> Dict:
    """
    Preprocess rainfall sequences AND create conditioning vectors.
    
    Parameters:
    -----------
    rainfall_scenarios : List[pd.DataFrame]
        List of 50 rainfall scenario DataFrames
    norm_method : str
        Normalization method for intensities
    info_csv_path : str, optional
        Path to info.csv with scenario metadata
        If None, conditioning vectors are NOT created
    
    Returns:
    --------
    Dict with:
        - sequences: Normalized rainfall (50, T)
        - conditioning_vectors: Scenario metadata (50, 4) - NEW
        - encoding_info: Conditioning encoding details - NEW
        - ... (other existing outputs)
    """
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
        
        if i <= 10 or i in [11, 20, 21, 30, 31, 50]:  # Sample output
            print(f"  RS{i:02d}: {len(intensity)} timesteps, "
                  f"peak={intensity.max():.2f} mm/hr at t={intensity.argmax()}")
        elif i == 11:
            print("  ...")
    
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
    # 3.2.5 CREATE CONDITIONING VECTORS (NEW)
    # -------------------------------------------------------------------------
    conditioning_vectors = None
    encoding_info = None
    
    if info_csv_path is not None:
        try:
            metadata_df = load_scenario_metadata(info_csv_path)
            conditioning_vectors, encoding_info = create_conditioning_vectors(metadata_df)
            print(f"\n✓ Conditioning vectors created")
        except Exception as e:
            print(f"\n⚠ WARNING: Could not create conditioning vectors: {e}")
            print(f"  Continuing without conditioning vectors...")
    else:
        print(f"\n⚠ No info.csv provided - conditioning vectors not created")
        print(f"  Model will not have explicit scenario metadata")
    
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
    
    # Temporal pattern preservation
    original_peaks = sequences.argmax(axis=1)
    normalized_peaks = sequences_norm.argmax(axis=1)
    peaks_preserved = np.array_equal(original_peaks, normalized_peaks)
    print(f"  Peak timing preserved: {'✓' if peaks_preserved else '✗'}")
    checks['peaks_preserved'] = peaks_preserved
    
    # Conditioning checks
    if conditioning_vectors is not None:
        cond_shape_ok = conditioning_vectors.shape == (num_scenarios, 4)
        cond_no_nans = not np.isnan(conditioning_vectors).any()
        print(f"  Conditioning shape: {conditioning_vectors.shape} == ({num_scenarios}, 4) {'✓' if cond_shape_ok else '✗'}")
        print(f"  Conditioning no NaNs: {'✓' if cond_no_nans else '✗'}")
        checks['conditioning_ok'] = cond_shape_ok and cond_no_nans
    
    # Statistics summary
    print(f"\n  Summary across {num_scenarios} scenarios:")
    print(f"    Mean peak intensity: {sequences_norm.max(axis=1).mean():.4f}")
    print(f"    Peak timing range: t={original_peaks.min()} to t={original_peaks.max()}")
    print(f"    Mean sequence sum: {sequences_norm.sum(axis=1).mean():.4f}")
    
    checks['all_passed'] = all(checks.values())
    
    results = {
        'sequences': sequences_norm,  # Shape: (num_scenarios, num_timesteps)
        'original_sequences': sequences,  # Keep for reference
        'conditioning_vectors': conditioning_vectors,  # NEW: (50, 4) or None
        'encoding_info': encoding_info,  # NEW: Encoding details
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
    if conditioning_vectors is not None:
        print(f"  Conditioning shape: {conditioning_vectors.shape}")
    print(f"  All checks passed: {checks['all_passed']}")
    print(f"{'='*70}\n")
    
    return results


def visualize_rainfall_sequences(results: Dict, num_to_plot: int = 20, figsize=(15, 10)):    
    """Existing visualization function - unchanged"""
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


def visualize_conditioning_vectors(results: Dict, figsize=(16, 10)):
    """
    NEW: Visualize conditioning vectors to understand scenario characteristics.
    """
    if results['conditioning_vectors'] is None:
        print("No conditioning vectors to visualize")
        return None
    
    cond = results['conditioning_vectors']
    encoding_info = results['encoding_info']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Pattern type distribution
    pattern_types = cond[:, 0]
    pattern_names = encoding_info['pattern_names']
    unique, counts = np.unique(pattern_types, return_counts=True)
    
    axes[0, 0].bar(unique, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0, 0].set_xticks(unique)
    axes[0, 0].set_xticklabels([pattern_names[int(u)] for u in unique], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Pattern Type Distribution', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Depth distribution
    depths = cond[:, 1] * encoding_info['max_depth_mm']
    axes[0, 1].hist(depths, bins=20, color='steelblue', edgecolor='black')
    axes[0, 1].set_xlabel('Total Depth (mm)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Rainfall Depth Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Depth by pattern type
    for i, name in enumerate(pattern_names):
        mask = pattern_types == i
        if mask.any():
            axes[1, 0].scatter(np.where(mask)[0] + 1, depths[mask], 
                             label=name, alpha=0.7, s=50)
    axes[1, 0].set_xlabel('Scenario ID')
    axes[1, 0].set_ylabel('Total Depth (mm)')
    axes[1, 0].set_title('Depth by Scenario', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # tpeak distribution (triangular only)
    triangular_mask = (cond[:, 3] == 1.0)
    if triangular_mask.any():
        tpeaks = cond[triangular_mask, 2]
        axes[1, 1].hist(tpeaks, bins=10, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Peak Time Fraction')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'tpeak Distribution (Triangular, n={triangular_mask.sum()})', 
                           fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Triangular Scenarios', 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('tpeak Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Conditioning Vector Analysis', fontsize=16, fontweight='bold', y=1.00)
    
    return fig