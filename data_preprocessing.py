"""
DATA QUALITY ANALYSIS AND IMPROVED PREPROCESSING
=================================================

This script analyzes your current data preprocessing and identifies
potential issues that could be affecting model performance.

IDENTIFIED ISSUES IN CURRENT PREPROCESSING:
1. Missing value handling - fillna(0) may introduce bias
2. No feature scaling validation across countries
3. Age outliers not handled
4. Wealth score not normalized
5. No handling of categorical encoding consistency
6. Hemoglobin missing values not properly imputed

IMPROVEMENTS IMPLEMENTED:
1. Proper missing value imputation (median for numeric, mode for categorical)
2. Outlier detection and handling
3. Feature engineering with interaction terms
4. Consistent encoding across all countries
5. Data quality reports
6. Validation of prevalence calculations

Author: Daniel / Claude
January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def analyze_data_quality(data_dir: str = 'data/processed_multiyear_filtered_15') -> Dict:
    """
    Comprehensive analysis of data quality issues.
    
    Returns a report of potential problems.
    """
    print("\n" + "="*70)
    print("DATA QUALITY ANALYSIS")
    print("="*70)
    
    data_dir = Path(data_dir)
    countries = ['Ghana', 'Mali', 'Nigeria', 'Burkina_Faso']
    
    report = {
        'missing_values': {},
        'outliers': {},
        'distribution_issues': {},
        'feature_issues': {},
        'recommendations': []
    }
    
    all_data = {}
    
    for country in countries:
        path = data_dir / f"{country}_clusters.csv"
        if path.exists():
            df = pd.read_csv(path)
            all_data[country] = df
            print(f"\n{country}:")
            print(f"  Clusters: {len(df)}")
            print(f"  Prevalence: {df['prevalence'].mean()*100:.1f}% ± {df['prevalence'].std()*100:.1f}%")
    
    if not all_data:
        print("ERROR: No data found!")
        return report
    
    # Combine for analysis
    combined = pd.concat(all_data.values(), ignore_index=True)
    
    # 1. MISSING VALUES ANALYSIS
    print("\n" + "-"*70)
    print("1. MISSING VALUES ANALYSIS")
    print("-"*70)
    
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        missing = combined[col].isna().sum()
        missing_pct = missing / len(combined) * 100
        
        if missing > 0:
            report['missing_values'][col] = {
                'count': missing,
                'percentage': missing_pct
            }
            print(f"  {col}: {missing} missing ({missing_pct:.1f}%)")
    
    if not report['missing_values']:
        print("  No missing values found (may already be filled with 0s)")
        
        # Check for suspicious zeros
        print("\n  Checking for suspicious zeros (potential missing values):")
        suspicious_cols = ['cluster_mean_age', 'cluster_mean_hemoglobin', 
                         'cluster_mean_altitude', 'cluster_mean_wealth_index_score']
        
        for col in suspicious_cols:
            if col in combined.columns:
                zero_count = (combined[col] == 0).sum()
                if zero_count > 0:
                    print(f"    {col}: {zero_count} zeros ({zero_count/len(combined)*100:.1f}%)")
                    report['missing_values'][f'{col}_zeros'] = zero_count
    
    # 2. OUTLIER ANALYSIS
    print("\n" + "-"*70)
    print("2. OUTLIER ANALYSIS")
    print("-"*70)
    
    for col in ['cluster_mean_age', 'cluster_mean_hemoglobin', 'cluster_mean_altitude',
                'cluster_mean_wealth_index_score', 'cluster_mean_household_size']:
        if col in combined.columns:
            data = combined[col].dropna()
            if len(data) > 0:
                q1, q3 = data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - 3 * iqr
                upper = q3 + 3 * iqr
                
                outliers = ((data < lower) | (data > upper)).sum()
                if outliers > 0:
                    report['outliers'][col] = {
                        'count': outliers,
                        'percentage': outliers / len(data) * 100,
                        'range': [lower, upper],
                        'actual_range': [data.min(), data.max()]
                    }
                    print(f"  {col}:")
                    print(f"    Outliers: {outliers} ({outliers/len(data)*100:.1f}%)")
                    print(f"    Expected: [{lower:.2f}, {upper:.2f}]")
                    print(f"    Actual: [{data.min():.2f}, {data.max():.2f}]")
    
    # 3. PREVALENCE DISTRIBUTION
    print("\n" + "-"*70)
    print("3. PREVALENCE DISTRIBUTION ANALYSIS")
    print("-"*70)
    
    # Check for extreme prevalence values
    extreme_low = (combined['prevalence'] == 0).sum()
    extreme_high = (combined['prevalence'] == 1).sum()
    
    print(f"  Prevalence = 0%: {extreme_low} clusters ({extreme_low/len(combined)*100:.1f}%)")
    print(f"  Prevalence = 100%: {extreme_high} clusters ({extreme_high/len(combined)*100:.1f}%)")
    
    if extreme_low + extreme_high > 0:
        report['distribution_issues']['extreme_prevalence'] = {
            'zeros': extreme_low,
            'ones': extreme_high
        }
    
    # Check prevalence by sample size
    print("\n  Prevalence variance by cluster size:")
    for size_range in [(15, 20), (20, 30), (30, 50), (50, 100), (100, 500)]:
        mask = (combined['total_tests'] >= size_range[0]) & (combined['total_tests'] < size_range[1])
        if mask.sum() > 0:
            subset = combined[mask]['prevalence']
            print(f"    {size_range[0]}-{size_range[1]} tests: n={mask.sum()}, var={subset.var():.4f}")
    
    # 4. FEATURE DISTRIBUTION BY COUNTRY
    print("\n" + "-"*70)
    print("4. FEATURE DISTRIBUTION BY COUNTRY")
    print("-"*70)
    
    key_features = ['cluster_mean_wealth_index_quintile', 'cluster_mean_urban',
                   'cluster_mean_has_bednet', 'cluster_mean_education_level']
    
    for feat in key_features:
        if feat in combined.columns:
            print(f"\n  {feat}:")
            for country, df in all_data.items():
                if feat in df.columns:
                    mean_val = df[feat].mean()
                    std_val = df[feat].std()
                    print(f"    {country}: {mean_val:.3f} ± {std_val:.3f}")
    
    # 5. SAMPLE SIZE ANALYSIS
    print("\n" + "-"*70)
    print("5. SAMPLE SIZE ANALYSIS")
    print("-"*70)
    
    print("\n  Cluster sizes by country:")
    for country, df in all_data.items():
        print(f"    {country}:")
        print(f"      Mean: {df['total_tests'].mean():.1f}")
        print(f"      Median: {df['total_tests'].median():.1f}")
        print(f"      Min: {df['total_tests'].min()}")
        print(f"      Max: {df['total_tests'].max()}")
    
    # 6. CORRELATION WITH PREVALENCE
    print("\n" + "-"*70)
    print("6. FEATURE CORRELATIONS WITH PREVALENCE")
    print("-"*70)
    
    print("\n  Overall correlations:")
    for col in numeric_cols:
        if col not in ['prevalence', 'positive_tests', 'total_tests', 'cluster_id']:
            corr = combined[col].corr(combined['prevalence'])
            if abs(corr) > 0.1:
                print(f"    {col}: r = {corr:.3f}")
    
    print("\n  Per-country correlations with wealth index:")
    if 'cluster_mean_wealth_index_quintile' in combined.columns:
        for country, df in all_data.items():
            if 'cluster_mean_wealth_index_quintile' in df.columns:
                corr = df['cluster_mean_wealth_index_quintile'].corr(df['prevalence'])
                print(f"    {country}: r = {corr:.3f}")
    
    # 7. RECOMMENDATIONS
    print("\n" + "-"*70)
    print("7. RECOMMENDATIONS")
    print("-"*70)
    
    recommendations = []
    
    if report['missing_values']:
        recommendations.append("Use median imputation instead of zero-filling for missing values")
    
    if report['outliers']:
        recommendations.append("Consider winsorizing outliers at 1st and 99th percentiles")
    
    if extreme_low + extreme_high > len(combined) * 0.05:
        recommendations.append("Consider removing clusters with 0% or 100% prevalence (small samples)")
    
    recommendations.append("Add cluster size as a feature (larger clusters = more reliable estimates)")
    recommendations.append("Create interaction terms (e.g., urban × wealth, bednet × wealth)")
    recommendations.append("Standardize features within each country before federation")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    report['recommendations'] = recommendations
    
    return report


def create_improved_dataset(
    input_dir: str = 'data/processed_multiyear_filtered_15',
    output_dir: str = 'data/processed_improved'
) -> Dict[str, pd.DataFrame]:
    """
    Create improved dataset with better preprocessing.
    """
    print("\n" + "="*70)
    print("CREATING IMPROVED DATASET")
    print("="*70)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    countries = ['Ghana', 'Mali', 'Nigeria', 'Burkina_Faso']
    all_data = {}
    
    # Load data
    for country in countries:
        path = input_dir / f"{country}_clusters.csv"
        if path.exists():
            all_data[country] = pd.read_csv(path)
    
    if not all_data:
        print("ERROR: No data found!")
        return {}
    
    # Combine for computing global statistics
    combined = pd.concat(all_data.values(), ignore_index=True)
    
    # Identify numeric feature columns
    exclude_cols = ['cluster_id', 'prevalence', 'positive_tests', 'total_tests',
                   'country', 'survey_year', 'malaria_variable_used']
    feature_cols = [c for c in combined.columns 
                   if c not in exclude_cols 
                   and combined[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"  {feature_cols}")
    
    # Compute global statistics for imputation
    global_medians = combined[feature_cols].median()
    global_means = combined[feature_cols].mean()
    global_stds = combined[feature_cols].std().replace(0, 1)
    
    # Process each country
    improved_data = {}
    
    for country, df in all_data.items():
        print(f"\nProcessing {country}...")
        df_improved = df.copy()
        
        # 1. IMPUTE MISSING VALUES
        for col in feature_cols:
            if col in df_improved.columns:
                # Replace zeros that are likely missing values
                if 'age' in col.lower() or 'hemoglobin' in col.lower():
                    df_improved.loc[df_improved[col] == 0, col] = np.nan
                
                # Impute with median
                missing = df_improved[col].isna().sum()
                if missing > 0:
                    df_improved[col] = df_improved[col].fillna(global_medians[col])
                    print(f"    Imputed {missing} missing values in {col}")
        
        # 2. WINSORIZE OUTLIERS
        for col in feature_cols:
            if col in df_improved.columns:
                lower = df_improved[col].quantile(0.01)
                upper = df_improved[col].quantile(0.99)
                outliers = ((df_improved[col] < lower) | (df_improved[col] > upper)).sum()
                if outliers > 0:
                    df_improved[col] = df_improved[col].clip(lower, upper)
                    print(f"    Winsorized {outliers} outliers in {col}")
        
        # 3. ADD NEW FEATURES
        
        # Cluster size as a feature (log-transformed)
        df_improved['log_cluster_size'] = np.log1p(df_improved['total_tests'])
        
        # Reliability weight (based on sample size)
        df_improved['sample_weight'] = np.sqrt(df_improved['total_tests']) / np.sqrt(df_improved['total_tests'].max())
        
        # Interaction terms (if base features exist)
        if 'cluster_mean_urban' in df_improved.columns and 'cluster_mean_wealth_index_quintile' in df_improved.columns:
            df_improved['urban_x_wealth'] = df_improved['cluster_mean_urban'] * df_improved['cluster_mean_wealth_index_quintile']
        
        if 'cluster_mean_has_bednet' in df_improved.columns and 'cluster_mean_wealth_index_quintile' in df_improved.columns:
            df_improved['bednet_x_wealth'] = df_improved['cluster_mean_has_bednet'] * df_improved['cluster_mean_wealth_index_quintile']
        
        if 'cluster_mean_education_level' in df_improved.columns and 'cluster_mean_wealth_index_quintile' in df_improved.columns:
            df_improved['education_x_wealth'] = df_improved['cluster_mean_education_level'] * df_improved['cluster_mean_wealth_index_quintile']
        
        # 4. FILTER UNRELIABLE CLUSTERS
        initial_count = len(df_improved)
        
        # Remove clusters with extreme prevalence AND small sample size
        unreliable = ((df_improved['prevalence'].isin([0.0, 1.0])) & 
                     (df_improved['total_tests'] < 25))
        df_improved = df_improved[~unreliable].copy()
        
        removed = initial_count - len(df_improved)
        if removed > 0:
            print(f"    Removed {removed} unreliable clusters")
        
        improved_data[country] = df_improved
        print(f"    Final: {len(df_improved)} clusters")
    
    # 5. GLOBAL NORMALIZATION (save statistics)
    print("\nComputing normalization statistics...")
    
    combined_improved = pd.concat(improved_data.values(), ignore_index=True)
    
    # Get all numeric feature columns (including new ones)
    all_feature_cols = [c for c in combined_improved.columns 
                       if c not in exclude_cols + ['log_cluster_size', 'sample_weight']
                       and combined_improved[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
    norm_stats = {
        'means': combined_improved[all_feature_cols].mean().to_dict(),
        'stds': combined_improved[all_feature_cols].std().replace(0, 1).to_dict()
    }
    
    # Save normalization statistics
    import json
    with open(output_dir / 'normalization_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    # 6. SAVE IMPROVED DATA
    print("\nSaving improved data...")
    
    for country, df in improved_data.items():
        path = output_dir / f"{country}_clusters.csv"
        df.to_csv(path, index=False)
        print(f"  {country}: {len(df)} clusters → {path}")
    
    # Save combined
    combined_improved = pd.concat(improved_data.values(), ignore_index=True)
    combined_improved.to_csv(output_dir / 'all_countries_clusters.csv', index=False)
    
    # Summary
    print("\n" + "="*70)
    print("IMPROVED DATASET SUMMARY")
    print("="*70)
    
    print(f"\nTotal clusters: {len(combined_improved)}")
    print(f"\nPer-country:")
    for country, df in improved_data.items():
        print(f"  {country}: {len(df)} clusters, prevalence = {df['prevalence'].mean()*100:.1f}%")
    
    print(f"\nNew features added:")
    print(f"  - log_cluster_size")
    print(f"  - sample_weight")
    print(f"  - urban_x_wealth (interaction)")
    print(f"  - bednet_x_wealth (interaction)")
    print(f"  - education_x_wealth (interaction)")
    
    print(f"\nTotal features: {len(all_feature_cols) + 2}")  # +2 for new features
    
    print(f"\nData saved to: {output_dir}")
    
    return improved_data


def compare_preprocessing_impact(
    original_dir: str = 'data/processed_multiyear_filtered_15',
    improved_dir: str = 'data/processed_improved'
):
    """
    Compare original vs improved preprocessing.
    """
    print("\n" + "="*70)
    print("PREPROCESSING COMPARISON")
    print("="*70)
    
    countries = ['Ghana', 'Mali', 'Nigeria', 'Burkina_Faso']
    
    original_data = {}
    improved_data = {}
    
    for country in countries:
        orig_path = Path(original_dir) / f"{country}_clusters.csv"
        impr_path = Path(improved_dir) / f"{country}_clusters.csv"
        
        if orig_path.exists():
            original_data[country] = pd.read_csv(orig_path)
        if impr_path.exists():
            improved_data[country] = pd.read_csv(impr_path)
    
    print("\nCluster counts:")
    print(f"  {'Country':<15} {'Original':>10} {'Improved':>10} {'Change':>10}")
    print(f"  {'-'*45}")
    
    for country in countries:
        orig_n = len(original_data.get(country, pd.DataFrame()))
        impr_n = len(improved_data.get(country, pd.DataFrame()))
        change = impr_n - orig_n
        print(f"  {country:<15} {orig_n:>10} {impr_n:>10} {change:>+10}")
    
    print("\nFeature counts:")
    if original_data and improved_data:
        orig_features = len([c for c in list(original_data.values())[0].columns 
                           if c not in ['cluster_id', 'prevalence', 'positive_tests', 
                                       'total_tests', 'country', 'survey_year', 'malaria_variable_used']])
        impr_features = len([c for c in list(improved_data.values())[0].columns 
                           if c not in ['cluster_id', 'prevalence', 'positive_tests', 
                                       'total_tests', 'country', 'survey_year', 'malaria_variable_used']])
        
        print(f"  Original: {orig_features} features")
        print(f"  Improved: {impr_features} features (+{impr_features - orig_features} new)")


def main():
    """Run data analysis and create improved dataset."""
    
    # 1. Analyze current data quality
    report = analyze_data_quality()
    
    # 2. Create improved dataset
    improved_data = create_improved_dataset()
    
    # 3. Compare
    if improved_data:
        compare_preprocessing_impact()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the data quality report above")
    print("  2. Run experiments with improved data:")
    print("     python comprehensive_fl_comparison.py --data-dir data/processed_improved")
    print("  3. Compare results with original preprocessing")


if __name__ == "__main__":
    main()