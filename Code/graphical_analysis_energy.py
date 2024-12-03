import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with actual data)
dysarthric_speakers = ['M01_1', 'M02_1', 'M04_2', 'F01_1', 'F03_1', 'F03_3', 'F03_3', 'F04_2']
control_speakers = ['MC01_1', 'MC02_1', 'MC03_1', 'MC03_2', 'MC04_1']

v_energy_dysarthria = [
    [35.6302062, 29.133821, 38.067097, 29.6376845, 34.5523718, 49.6781578, 39.4347182, 44.193169],
    [51.361548, 27.394205, 38.201478, 50.816972, 33.213289, 62.100626, 38.078234, 44.410957],
    [32.3699005, 28.66934425, 36.224976, 31.3049665, 33.4733615, 53.99986675, 41.609858, 48.56657625],
    [33.096481, 29.3475212, 37.1752876, 30.2449842, 29.7843934, 48.5508702, 38.1958284, 44.1391828],
]

v_energy_non_dysarthria = [
    [45.2544806, 45.833602, 58.0668592, 27.4248626, 40.3133708],
    [44.630142, 46.71, 58.418277, 28.877201, 39.722176],
    [44.52448325, 44.8036435, 57.59499175, 27.993773, 39.63849675],
    [44.5760786, 46.9029666, 57.8017664, 27.4403062, 41.6744952],
]

c_energy_dysarthria = [
    [34.689709, 29.5437975, 38.68722625, 28.3137075, 34.04624125, 52.66626525, 39.462963, 48.21966525],
    [37.76337767, 28.91117133, 36.36958133, 46.13354533, 33.83443, 49.425321, 42.67962067, 43.00132633],
    [39.98575567, 30.423038, 38.398962, 29.717101, 32.04682933, 59.03925533, 37.83111267, 51.20532633],
    [38.97926233, 28.33118967, 37.080969, 28.06964933, 31.812324, 57.64624, 36.924963, 43.73863833],
    [32.88217275, 28.62004275, 38.92023825, 27.3796945, 33.09596, 51.59239975, 39.739481, 44.23413325],
    [43.122323, 29.29392675, 36.121188, 29.07095833, 29.0996325, 53.93920267, 42.2402025, 43.892581],
    [28.9941, 29.537724, 38.295721, 29.373145, 33.346009, 62.612459, 40.038226, 43.381493],
    [26.706884, 26.666369, 37.775046, 31.649086, 27.369407, 61.123503, 38.259215, 40.978433],
    [30.170689, 29.197813, 38.929174, 29.372894, 31.623062, 60.940687, 41.739709, 44.922782],
    [34.044351, 28.64017014, 40.42267757, 31.123687, 27.226273, 54.92576129, 39.47783386, 43.80493886]
]


c_energy_non_dysarthria = [
    [45.45335125, 45.482284, 58.40807925, 27.97395975, 41.383793],
    [44.68533433, 45.79553567, 58.03406167, 29.21603167, 39.31037333],
    [45.04970467, 43.075894, 57.57566433, 27.02097333, 41.434968],
    [44.232456, 44.80499133, 58.72416567, 28.27216533, 39.78857267],
    [44.61410075, 47.49584325, 59.29031525, 27.21289125, 41.48790825],
    [43.2717, 45.8655095, 58.9489995, 28.816126, 37.348259],
    [45.19885, 46.242732, 56.270416, 26.260005, 40.382572],
    [45.051041, 47.161358, 52.997603, 27.156162, 42.463502],
    [46.353282, 47.224519, 58.31698, 28.835183, 40.103673],
    [45.08277814, 46.18485214, 58.596701, 26.99864486, 42.02319471]
]

# Define colors for vowels and consonants
vowel_color = 'red'
consonant_color = 'blue'
avg_vowel_color = 'red'
avg_consonant_color = 'blue'

# Create figure with two separate subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'hspace': 0.4})

# Calculate averages for dysarthric speakers
v_avg_dysarthria = np.mean(v_energy_dysarthria, axis=0)
c_avg_dysarthria = np.mean(c_energy_dysarthria, axis=0)

# Calculate averages for control speakers
v_avg_non_dysarthria = np.mean(v_energy_non_dysarthria, axis=0)
c_avg_non_dysarthria = np.mean(c_energy_non_dysarthria, axis=0)

# Find overall min and max values for y-axis
all_data = []
for data_list in v_energy_dysarthria + c_energy_dysarthria + v_energy_non_dysarthria + c_energy_non_dysarthria:
    all_data.extend(data_list)
y_min = min(all_data)
y_max = max(all_data)

# Plot Control Speakers (CTL) in ax1
x1 = np.arange(len(control_speakers))
for i, v_data in enumerate(v_energy_non_dysarthria):
    ax1.plot(x1, v_data, linestyle='--', marker='o', color=vowel_color, alpha=0.2, label=f'V{i+1}')

for i, c_data in enumerate(c_energy_non_dysarthria):
    ax1.plot(x1, c_data, linestyle='-', marker='s', color=consonant_color, alpha=0.2, label=f'C{i+1}')

# Add average lines for control speakers
ax1.plot(x1, v_avg_non_dysarthria, linestyle='-', linewidth=3, color=avg_vowel_color, label='Vowel (overall)')
ax1.plot(x1, c_avg_non_dysarthria, linestyle='-', linewidth=3, color=avg_consonant_color, label='Consonant (overall)')

ax1.set_title('Control (CTL) Speakers')
ax1.set_ylabel('Energy/Loudness (in dB)')
ax1.set_xticks(x1)
ax1.set_xticklabels(control_speakers, rotation=45)
ax1.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax1.grid(True)
ax1.set_ylim(y_min, y_max)  # Set y-axis limits

# Plot Dysarthric Speakers (DYS) in ax2
x2 = np.arange(len(dysarthric_speakers))
for i, v_data in enumerate(v_energy_dysarthria):
    ax2.plot(x2, v_data, linestyle='--', marker='o', color=vowel_color, alpha=0.2, label=f'V{i+1}')

for i, c_data in enumerate(c_energy_dysarthria):
    ax2.plot(x2, c_data, linestyle='-', marker='s', color=consonant_color, alpha=0.2, label=f'C{i+1}')

# Add average lines for dysarthric speakers
ax2.plot(x2, v_avg_dysarthria, linestyle='-', linewidth=3, color=avg_vowel_color, label='Vowel (overall)')
ax2.plot(x2, c_avg_dysarthria, linestyle='-', linewidth=3, color=avg_consonant_color, label='Consonant (overall)')

ax2.set_title('Dysarthric (DYS) Speakers')
ax2.set_ylabel('Energy/Loudness (in dB)')
ax2.set_xlabel('Speakers')
ax2.set_xticks(x2)
ax2.set_xticklabels(dysarthric_speakers, rotation=45)
ax2.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
ax2.grid(True)
ax2.set_ylim(y_min, y_max)  # Set y-axis limits

# Adjust layout to prevent overlap
plt.subplots_adjust(right=0.85)
plt.savefig("phonetic_analysis_energy.png")
plt.show()