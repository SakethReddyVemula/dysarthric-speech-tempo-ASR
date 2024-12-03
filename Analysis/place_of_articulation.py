import matplotlib.pyplot as plt
import numpy as np

# Data from the table
places = [
    "Bilabial", "Labiodental", "Dental", "Alveolar", "Post-alveolar",
    "Palatal", "Velar", "Glottal", "Average"
]
dysarthric_speakers = [
    0.278042667, 0.304992833, 0.284252833, 0.295834714, 0.297903944,
    0.262909, 0.287948333, 0.282917778, 0.289442567
]
controlled_speakers = [
    0.19193465, 0.1788471, 0.1752311, 0.186117171, 0.18789095,
    0.1587426, 0.173535267, 0.172931556, 0.181917047
]
tempo_ratios = [
    1.448631952, 1.705327251, 1.622159727, 1.5895079, 1.585515132,
    1.656196887, 1.659307292, 1.632390207, 1.612379543
]

# Bar plot for Dysarthric vs Controlled speakers
x = np.arange(len(places))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(x - width/2, dysarthric_speakers, width, label="Dysarthric Speakers", color='blue')
ax1.bar(x + width/2, controlled_speakers, width, label="Controlled Speakers", color='green')

# Add annotations and labels
ax1.set_xlabel("Place of Articulation")
ax1.set_ylabel("Proportions")
ax1.set_title("Dysarthric vs Controlled Speakers by Place of Articulation")
ax1.set_xticks(x)
ax1.set_xticklabels(places, rotation=45, ha='right')
ax1.legend()

# Line plot for Tempo-ratios
ax2 = ax1.twinx()
ax2.plot(x, tempo_ratios, label="Tempo Ratios", color='red', marker='o', linestyle='--')
ax2.set_ylabel("Tempo Ratios")
ax2.legend(loc="upper right")

# Show the plot
plt.tight_layout()
plt.show()
