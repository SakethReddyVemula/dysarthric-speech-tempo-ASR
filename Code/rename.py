import os

# Path to the directory containing the files
directory = '../Dataset/M01/Session1/phn_headMic/'

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file has a .PHN extension
    if filename.endswith('.phn'):
        # Separate the file name into the number and extension
        number_part = filename.split('.')[0]
        # Format the number part to be 4 digits with leading zeros
        new_number_part = number_part.zfill(4)
        # Construct the new filename
        new_filename = f"{new_number_part}.PHN"
        # Rename the file
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f"Renamed '{filename}' to '{new_filename}'")
