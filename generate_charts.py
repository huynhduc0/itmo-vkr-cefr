import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

def generate_visualizations():
    print("Loading UniversalCEFR/cefr_sp_en dataset...")
    try:
        ds = load_dataset('UniversalCEFR/cefr_sp_en', split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    os.makedirs('visuals', exist_ok=True)
    
    # Extract data
    levels = [x['cefr_level'] for x in ds]
    lens = [len(x['text'].split()) for x in ds]
    
    # 1. Class Distribution Chart
    plt.figure(figsize=(10, 6))
    sns.countplot(x=levels, order=['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], palette='viridis')
    plt.title('Original Data Distribution: UniversalCEFR/cefr_sp_en', fontsize=14)
    plt.xlabel('CEFR Level', fontsize=12)
    plt.ylabel('Number of Samples (Train Split)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path_1 = 'visuals/class_distribution.png'
    plt.savefig(output_path_1, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path_1}")
    plt.close()

    # 2. Token Length Distribution Chart
    plt.figure(figsize=(10, 6))
    sns.histplot(lens, bins=50, kde=True, color='teal')
    plt.title('Token Length Distribution (UniversalCEFR)', fontsize=14)
    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, 200)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path_2 = 'visuals/length_distribution.png'
    plt.savefig(output_path_2, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path_2}")
    plt.close()

if __name__ == "__main__":
    generate_visualizations()
