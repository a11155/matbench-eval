import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_and_prepare_data(prediction_files, ground_truth_file):
    """
    Load and prepare data, matching model2's predictions to model1's materials
    """
    # Load ground truth data
    df_ground_truth = pd.read_csv(ground_truth_file)
    
    # Load model1 data first to get reference material IDs
    df_model1 = pd.read_csv(prediction_files[0][0])
    reference_materials = set(df_model1['material_id'].tolist())
    print(f"\nNumber of materials in model1: {len(reference_materials)}")
    
    # Load and filter model2 data
    df_model2 = pd.read_csv(prediction_files[1][0])
    df_model2_filtered = df_model2[df_model2['material_id'].isin(reference_materials)]
    print(f"Original number of materials in model2: {len(df_model2)}")
    print(f"Number of materials in model2 after filtering: {len(df_model2_filtered)}")
    
    # Create the final DataFrame
    final_df = pd.DataFrame({
        'material_id': df_ground_truth['material_id'],
        'ground_truth_energy': df_ground_truth['eqV2-86M-omat-mp-salex_energy'],
        'ground_truth_form_energy': df_ground_truth['e_form_per_atom_eqV2-86M-omat-mp-salex']
    })
    
    # Add model1 data
    temp_df1 = pd.DataFrame({
        'material_id': df_model1['material_id'],
        'model1_energy': df_model1['sevennet_energy'],
        'model1_form_energy': df_model1['e_form_per_atom_sevennet']
    })
    final_df = pd.merge(final_df, temp_df1, on='material_id', how='outer')
    
    # Add filtered model2 data
    temp_df2 = pd.DataFrame({
        'material_id': df_model2_filtered['material_id'],
        'model2_energy': df_model2_filtered['sevennet_energy'],
        'model2_form_energy': df_model2_filtered['e_form_per_atom_sevennet']
    })
    final_df = pd.merge(final_df, temp_df2, on='material_id', how='outer')
    
    return final_df

def calculate_metrics(df, model_name):
    """
    Calculate metrics for a model
    """
    mask = pd.notna(df[f'{model_name}_energy']) & pd.notna(df['ground_truth_energy'])
    df_valid = df[mask]
    
    if len(df_valid) == 0:
        print(f"Warning: No valid comparisons found for {model_name}")
        return None
    
    # Calculate metrics
    energy_diff = df_valid[f'{model_name}_energy'] - df_valid['ground_truth_energy']
    form_energy_diff = df_valid[f'{model_name}_form_energy'] - df_valid['ground_truth_form_energy']
    
    metrics = {
        'model': model_name,
        'n_materials': len(df_valid),
        'energy_mae': np.abs(energy_diff).mean(),
        'energy_rmse': np.sqrt((energy_diff ** 2).mean()),
        'form_energy_mae': np.abs(form_energy_diff).mean(),
        'form_energy_rmse': np.sqrt((form_energy_diff ** 2).mean()),
        'energy_correlation': stats.pearsonr(df_valid['ground_truth_energy'], 
                                           df_valid[f'{model_name}_energy'])[0],
        'form_energy_correlation': stats.pearsonr(df_valid['ground_truth_form_energy'], 
                                                df_valid[f'{model_name}_form_energy'])[0]
    }
    
    return metrics

def plot_comparisons(df, model_names):
    """
    Create comparison plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red']
    markers = ['o', 's']
    
    # Energy comparison plots
    for idx, model_name in enumerate(model_names):
        mask = pd.notna(df[f'{model_name}_energy']) & pd.notna(df['ground_truth_energy'])
        df_valid = df[mask]
        
        ax1.scatter(
            df_valid['ground_truth_energy'],
            df_valid[f'{model_name}_energy'],
            color=colors[idx],
            marker=markers[idx],
            alpha=0.6,
            label=f'{model_name} (n={len(df_valid)})'
        )
    
    # Add diagonal line for energy plot
    energy_min = df['ground_truth_energy'].min()
    energy_max = df['ground_truth_energy'].max()
    ax1.plot([energy_min, energy_max], [energy_min, energy_max], 'k--')
    ax1.set_xlabel('Ground Truth Energy')
    ax1.set_ylabel('Predicted Energy')
    ax1.set_title('Energy Comparison')
    ax1.legend()
    
    # Formation energy comparison plots
    for idx, model_name in enumerate(model_names):
        mask = pd.notna(df[f'{model_name}_form_energy']) & pd.notna(df['ground_truth_form_energy'])
        df_valid = df[mask]
        
        ax2.scatter(
            df_valid['ground_truth_form_energy'],
            df_valid[f'{model_name}_form_energy'],
            color=colors[idx],
            marker=markers[idx],
            alpha=0.6,
            label=f'{model_name} (n={len(df_valid)})'
        )
    
    # Add diagonal line for formation energy plot
    form_min = df['ground_truth_form_energy'].min()
    form_max = df['ground_truth_form_energy'].max()
    ax2.plot([form_min, form_max], [form_min, form_max], 'k--')
    ax2.set_xlabel('Ground Truth Formation Energy per Atom')
    ax2.set_ylabel('Predicted Formation Energy per Atom')
    ax2.set_title('Formation Energy Comparison')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('energy_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # File paths and model names
    prediction_files = [
        ('checkpoint_best.pth.csv.gz', 'model1'),
        ('2024-07-11-sevennet-0-preds.csv.gz', 'model2')
    ]
    ground_truth_file = 'eqV2-m-omat-mp-salex.csv.gz'
    
    # Load and prepare data
    merged_df = load_and_prepare_data(prediction_files, ground_truth_file)
    
    # Calculate metrics for each model
    all_metrics = []
    model_names = ['model1', 'model2']
    for model_name in model_names:
        metrics = calculate_metrics(merged_df, model_name)
        if metrics is not None:
            all_metrics.append(metrics)
    
    if all_metrics:
        # Create comparison DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Print results
        print("\nAnalysis Results:")
        print("-" * 80)
        for model_name in model_names:
            model_metrics = metrics_df[metrics_df['model'] == model_name]
            if not model_metrics.empty:
                model_metrics = model_metrics.iloc[0]
                print(f"\nModel: {model_name}")
                print(f"Number of materials compared: {model_metrics['n_materials']}")
                
                print("\nEnergy Metrics:")
                print(f"MAE: {model_metrics['energy_mae']:.4f}")
                print(f"RMSE: {model_metrics['energy_rmse']:.4f}")
                print(f"Correlation: {model_metrics['energy_correlation']:.4f}")
                
                print("\nFormation Energy per Atom Metrics:")
                print(f"MAE: {model_metrics['form_energy_mae']:.4f}")
                print(f"RMSE: {model_metrics['form_energy_rmse']:.4f}")
                print(f"Correlation: {model_metrics['form_energy_correlation']:.4f}")
        
        # Create plots
        plot_comparisons(merged_df, model_names)
        
        # Save detailed results
        merged_df.to_csv('detailed_comparison.csv', index=False)
        metrics_df.to_csv('model_metrics.csv', index=False)

if __name__ == "__main__":
    main()
