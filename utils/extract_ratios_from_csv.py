import pandas as pd
import json

path = "../Dysarthria_Phonetic_Analysis(Ratios).csv"
df = pd.read_csv(path)

phoneme_dict = dict(zip(df["Phoneme"], df["F01_S1_Ratios"]))
# Save the dictionary as a JSON file
output_file = "F01_S1.json"  # Replace with your desired output file name
with open(output_file, 'w') as json_file:
    json.dump(phoneme_dict, json_file, indent=4)

print(f"Dictionary saved as {output_file}")