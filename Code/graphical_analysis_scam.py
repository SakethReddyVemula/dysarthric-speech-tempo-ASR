import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with actual data)
dysarthric_speakers = ['M01_1', 'M02_1', 'M04_2', 'F01_1', 'F03_1', 'F03_3', 'F03_3', 'F04_2']
control_speakers = ['MC01_1', 'MC02_1', 'MC03_1', 'MC03_2', 'MC04_1']

v_durations_dysarthria = [
    [0.289243, 0.245706, 0.3112828, 0.23769175, 0.2313322, 0.2226722, 0.1467992, 0.245123],
    [0.351345, 0.225335, 0.314967, 0.393249, 0.20048, 0.334261, 0.126721, 0.232949],
    [0.31131625, 0.2332365, 0.33204375,  0.2638525, 0.24869925, 0.235319, 0.15416025, 0.2434295],
    [0.33226, 0.2374842, 0.3186618, 0.2296774, 0.199114, 0.2553218, 0.1504784, 0.244989]
]

v_durations_non_dysartria = [
    [0.2346494, 0.2258606, 0.1478198, 0.125256, 0.152976],
    [0.236049, 0.219447, 0.152523, 0.138117, 0.15344],
    [0.24365725, 0.22851625, 0.15961975, 0.1299785, 0.15943675],
    [0.2667736, 0.2380542, 0.1613434, 0.1340694, 0.1675118]
]

c_durations_dysarthria = [
    [0.2982835, 0.19518675, 0.295218, 0.1966165, 0.18917125, 0.204038, 0.13389725, 0.21329525],
    [0.253068333, 0.222541333, 0.306423, 0.240317667, 0.180728333, 0.193693, 0.117764, 0.200915],
    [0.275266, 0.185504667, 0.278505333, 0.216209333, 0.172345, 0.247172667, 0.11891, 0.200723333],
    [0.279916667, 0.189946, 0.288490667, 0.209364667, 0.184378, 0.206984667, 0.113659667, 0.207836667],
    [0.340613333, 0.20806225, 0.29172975, 0.22367725, 0.17533575, 0.21002675, 0.105024, 0.212402],
    [0.2881355, 0.1683145, 0.274797333, 0.227838, 0.17725175, 0.2881355, 0.12132575, 0.213749333],
    [0.326619, 0.216282, 0.303796, 0.195251, 0.139305, 0.245955, 0.10594, 0.221167],
    [0.345198, 0.191309, 0.287298, 0.262138, 0.142665, 0.194992, 0.126993, 0.171397],
    [0.305991, 0.198285, 0.243261, 0.191392, 0.179625, 0.199982, 0.158721, 0.197462]
]


c_durations_non_dysarthria = [
    [0.2232255, 0.20351125, 0.12934025, 0.092786, 0.130561],
    [0.228825333, 0.216923667, 0.131803333, 0.101644333, 0.132734333],
    [0.210316, 0.2380185, 0.122919333, 0.097755333, 0.129120667],
    [0.195897, 0.186784667, 0.11687, 0.091144667, 0.120167667],
    [0.238057, 0.23347625, 0.14635875, 0.1167135, 0.1474915],
    [0.150067, 0.1329025, 0.09274975, 0.077442, 0.0956485],
    [0.293033, 0.286889, 0.164659, 0.146668, 0.155771],
    [0.20045, 0.185699, 0.110222, 0.06461, 0.121044],
    [0.220456, 0.194975, 0.126638, 0.126279, 0.129857]
]


# Define colors for vowels and consonants
vowel_color = 'red'
consonant_color = 'blue'
avg_vowel_color = 'red'
avg_consonant_color = 'blue'

# Create figure with two separate subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'hspace': 0.4})

# Calculate averages for dysarthric speakers
v_avg_dysarthria = np.mean(v_durations_dysarthria, axis=0)
c_avg_dysarthria = np.mean(c_durations_dysarthria, axis=0)

# Calculate averages for control speakers
v_avg_non_dysarthria = np.mean(v_durations_non_dysartria, axis=0)
c_avg_non_dysarthria = np.mean(c_durations_non_dysarthria, axis=0)

# Find overall min and max values for y-axis
all_data = []
for data_list in v_durations_dysarthria + c_durations_dysarthria + v_durations_non_dysartria + c_durations_non_dysarthria:
    all_data.extend(data_list)
y_min = min(all_data)
y_max = max(all_data)

# Plot Control Speakers (CTL) in ax1
x1 = np.arange(len(control_speakers))
for i, v_data in enumerate(v_durations_non_dysartria):
    ax1.plot(x1, v_data, linestyle='--', marker='o', color=vowel_color, alpha=0.2, label=f'V{i+1}')

for i, c_data in enumerate(c_durations_non_dysarthria):
    ax1.plot(x1, c_data, linestyle='-', marker='s', color=consonant_color, alpha=0.2, label=f'C{i+1}')

# Add average lines for control speakers
ax1.plot(x1, v_avg_non_dysarthria, linestyle='-', linewidth=3, color=avg_vowel_color, label='Vowel (overall)')
ax1.plot(x1, c_avg_non_dysarthria, linestyle='-', linewidth=3, color=avg_consonant_color, label='Consonant (overall)')

ax1.set_title('Control (CTL) Speakers')
ax1.set_ylabel('Duration (seconds)')
ax1.set_xticks(x1)
ax1.set_xticklabels(control_speakers, rotation=45)
ax1.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax1.grid(True)
ax1.set_ylim(y_min, y_max)  # Set y-axis limits

# Plot Dysarthric Speakers (DYS) in ax2
x2 = np.arange(len(dysarthric_speakers))
for i, v_data in enumerate(v_durations_dysarthria):
    ax2.plot(x2, v_data, linestyle='--', marker='o', color=vowel_color, alpha=0.2, label=f'V{i+1}')

for i, c_data in enumerate(c_durations_dysarthria):
    ax2.plot(x2, c_data, linestyle='-', marker='s', color=consonant_color, alpha=0.2, label=f'C{i+1}')

# Add average lines for dysarthric speakers
ax2.plot(x2, v_avg_dysarthria, linestyle='-', linewidth=3, color=avg_vowel_color, label='Vowel (overall)')
ax2.plot(x2, c_avg_dysarthria, linestyle='-', linewidth=3, color=avg_consonant_color, label='Consonant (overall)')

ax2.set_title('Dysarthric (DYS) Speakers')
ax2.set_ylabel('Duration (seconds)')
ax2.set_xlabel('Speakers')
ax2.set_xticks(x2)
ax2.set_xticklabels(dysarthric_speakers, rotation=45)
ax2.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax2.grid(True)
ax2.set_ylim(y_min, y_max)  # Set y-axis limits

# Adjust layout to prevent overlap
plt.subplots_adjust(right=0.85)
plt.savefig("phonetic_analysis_scam.png")
plt.show()