import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the data
data = {
    'Place of Articulation': ['Bilabial', 'Labiodental', 'Dental', 'Alveolar', 
                            'Post-alveolar', 'Palatal', 'Velar', 'Glottal'],
    'M01': [1.50470833, 2.15375871, 1.85472784, 1.77623359, 
            1.85261052, 1.73500371, 1.78860090, 1.81761965],
    'M02': [1.20246831, 1.26674405, 1.26097194, 1.22123452, 
            1.16836788, 1.29181455, 1.32307785, 1.30067773],
    'M04': [1.63871922, 1.69547899, 1.75077940, 1.77105559, 
            1.73556701, 1.94177240, 1.86624313, 1.77883093]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create the line plot
plt.figure(figsize=(12, 6))

# Plot lines for each speaker with different styles and markers
plt.plot(df['Place of Articulation'], df['M01'], marker='o', label='M01', linewidth=2)
plt.plot(df['Place of Articulation'], df['M02'], marker='s', label='M02', linewidth=2)
plt.plot(df['Place of Articulation'], df['M04'], marker='^', label='M04', linewidth=2)

# Customize the plot
plt.title('Speech Tempo Ratios by Place of Articulation', fontsize=14, pad=20)
plt.xlabel('Place of Articulation', fontsize=12)
plt.ylabel('Speech Tempo Ratio', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(title='Speakers', bbox_to_anchor=(1.02, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()