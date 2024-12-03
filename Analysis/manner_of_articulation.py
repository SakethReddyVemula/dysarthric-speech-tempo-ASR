import matplotlib.pyplot as plt
import numpy as np

# Data from the second table
manners = [
    "Long Vowels", "Medium Vowels", "Short Vowels", "Dipthongs", "Glides",
    "Liquids", "Nasals", "Unvoiced Fricatives", "Voiced Fricatives",
    "Unvoiced Affricatives", "Voiced Affricatives", "Unvoiced Stops",
    "Voiced Stops", "Aspirates", "Average"
]
dysarthric_speakers = [
    0.282077267, 0.297215667, 0.292198833, 0.296135333, 0.292896083,
    0.290677556, 0.276425333, 0.282784444, 0.310135111, 0.273749111,
    0.312232333, 0.304601667, 0.279179, 0.290719714, 0.291501961
]
controlled_speakers = [
    0.17731236, 0.1799152, 0.1842417, 0.19355048, 0.1858848,
    0.1923862, 0.189625967, 0.1721728, 0.2064194, 0.13976195,
    0.239404, 0.166405, 0.189641, 0.217946525, 0.188190527
]
tempo_ratios = [
    1.590849429, 1.651976413, 1.585953849, 1.530016011, 1.575686034,
    1.510906476, 1.457739877, 1.642445522, 1.502451374, 1.958681251,
    1.304206836, 1.83048386, 1.472144737, 1.333903875, 1.567674682
]

# Bar plot for Dysarthric vs Controlled speakers
x = np.arange(len(manners))
width = 0.35

fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.bar(x - width/2, dysarthric_speakers, width, label="Dysarthric Speakers", color='blue')
ax1.bar(x + width/2, controlled_speakers, width, label="Controlled Speakers", color='green')

# Add annotations and labels
ax1.set_xlabel("Manner of Articulation")
ax1.set_ylabel("Proportions")
ax1.set_title("Dysarthric vs Controlled Speakers by Manner of Articulation")
ax1.set_xticks(x)
ax1.set_xticklabels(manners, rotation=45, ha='right')
ax1.legend()

# Line plot for Tempo-ratios
ax2 = ax1.twinx()
ax2.plot(x, tempo_ratios, label="Tempo Ratios", color='red', marker='o', linestyle='--')
ax2.set_ylabel("Tempo Ratios")
ax2.legend(loc="upper left")

# Show the plot
plt.tight_layout()
plt.show()
