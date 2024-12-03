import pandas as pd

# Specify the input CSV file and output Excel file paths
csv_file = "/media/saketh/New Volume/SAL/ASR/M04_results_cleaned/M04_Session2_wer_results.csv"  # Replace with your CSV file path
excel_file = "/media/saketh/New Volume/SAL/ASR/M04_results_model_2/M04_Session2_wer_results.xlsx"  # Replace with your desired Excel file path

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Write the DataFrame to an Excel file
df.to_excel(excel_file, index=False)

print(f"CSV file has been successfully converted to Excel format: {excel_file}")
