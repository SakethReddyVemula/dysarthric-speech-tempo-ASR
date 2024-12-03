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
    [0.3282835, 0.22518675, 0.325218, 0.2266165, 0.21917125, 0.234038, 0.16389725, 0.24329525],
    [0.283068333, 0.252541333, 0.336423, 0.270317667, 0.210728333, 0.223693, 0.147764, 0.230915],
    [0.305266, 0.215504667, 0.308505333, 0.246209333, 0.202345, 0.277172667, 0.14891, 0.230723333],
    [0.309916667, 0.219946, 0.318490667, 0.239364667, 0.214378, 0.236984667, 0.143659667, 0.237836667],
    [0.370613333, 0.23806225, 0.32172975, 0.25367725, 0.20533575, 0.24002675, 0.135024, 0.242402],
    [0.3181355, 0.1983145, 0.304797333, 0.257838, 0.20725175, 0.3181355, 0.15132575, 0.243749333],
    [0.356619, 0.246282, 0.333796, 0.225251, 0.169305, 0.275955, 0.13594, 0.251167],
    [0.375198, 0.221309, 0.317298, 0.292138, 0.172665, 0.224992, 0.156993, 0.201397],
    [0.335991, 0.228285, 0.273261, 0.221392, 0.209625, 0.229982, 0.188721, 0.227462]
]

c_durations_non_dysarthria = [
    [0.2532255, 0.23351125, 0.15934025, 0.122786, 0.160561],
    [0.258825333, 0.246923667, 0.161803333, 0.131644333, 0.162734333],
    [0.240316, 0.2680185, 0.152919333, 0.127755333, 0.159120667],
    [0.225897, 0.216784667, 0.14687, 0.121144667, 0.150167667],
    [0.268057, 0.26347625, 0.17635875, 0.1467135, 0.1774915],
    [0.180067, 0.1629025, 0.12274975, 0.107442, 0.1256485],
    [0.323033, 0.316889, 0.194659, 0.176668, 0.185771],
    [0.23045, 0.215699, 0.140222, 0.09461, 0.151044],
    [0.250456, 0.224975, 0.156638, 0.156279, 0.159857]
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
plt.savefig("phonetic_analysis.png")
plt.show()