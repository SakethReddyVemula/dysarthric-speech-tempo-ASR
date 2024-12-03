import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the data
data = {
    'Manner of Articulation': [
        'Long Vowels', 'Medium Vowels', 'Short Vowels', 'Dipthongs', 'Glides',
        'Liquids', 'Nasals', 'Unvoiced Fricatives', 'Voiced Fricatives',
        'Unvoiced Affricatives', 'Voiced Affricatives', 'Unvoiced Stops',
        'Voiced Stops', 'Aspirates'
    ],
    'M01': [1.631262479, 1.95283667, 1.689716552, 1.716658104, 1.766058871,
            1.471354668, 1.609832268, 1.80003268, 1.795438478, 2.276266895,
            1.489611702, 2.254727923, 1.771721305, 1.359688706],
    'M02': [1.385724041, 1.252451155, 1.265926769, 1.226988432, 1.211431758,
            1.312679045, 1.136472343, 1.2774724, 1.153293973, 1.418944856,
            1.028729679, 1.329942009, 1.2037745, 1.117677179],
    'M04': [1.755561767, 1.750641413, 1.802218227, 1.646401497, 1.749567474,
            1.748685717, 1.62691502, 1.849831487, 1.55862167, 2.180832003,
            1.394279126, 1.906781647, 1.440938405, 1.524345741]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create the line plot
plt.figure(figsize=(15, 8))

# Plot lines for each speaker with different styles and markers
plt.plot(df['Manner of Articulation'], df['M01'], marker='o', label='M01', linewidth=2)
plt.plot(df['Manner of Articulation'], df['M02'], marker='s', label='M02', linewidth=2)
plt.plot(df['Manner of Articulation'], df['M04'], marker='^', label='M04', linewidth=2)

# Customize the plot
plt.title('Speech Tempo Ratios by Manner of Articulation', fontsize=14, pad=20)
plt.xlabel('Manner of Articulation', fontsize=12)
plt.ylabel('Speech Tempo Ratio', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(title='Speakers', bbox_to_anchor=(1.02, 1), loc='upper left')

# Set y-axis limits to better show the variations
plt.ylim(1.0, 2.5)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()