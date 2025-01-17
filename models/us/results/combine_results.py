import json
import gzip
import glob
from pathlib import Path
# Example usage
def merge_json_files(input_pattern: str, output_file: str):
    """
    Merge multiple .json.gz files into a single file with sequential indexing.
    """
    merged_data = {
        "material_id": {},
        "sevennet_structure": {},
        "sevennet_energy": {}
    }

    next_index = 0

    for file_path in glob.glob(input_pattern):
        print(file_path)
        with gzip.open(file_path, 'rt') as f:
            data = json.load(f)

            # Get number of entries in this file
            num_entries = len(data["material_id"])

            for i in range(num_entries):
                str_i = str(i)  # Original index in file
                str_next = str(next_index)  # New index in merged file

                # Copy each entry with new index
                merged_data["material_id"][str_next] = data["material_id"][str_i]
                merged_data["sevennet_structure"][str_next] = data["sevennet_structure"][str_i]
                merged_data["sevennet_energy"][str_next] = data["sevennet_energy"][str_i]

                next_index += 1

    # Save merged data
    output_path = Path(output_file)
    if output_file.endswith('.gz'):
        with gzip.open(output_path, 'wt') as f:
            json.dump(merged_data, f)
    else:
        with open(output_path, 'w') as f:
            json.dump(merged_data, f)

if __name__ == "__main__":
    merge_json_files("predictions_epoch600/*.json.gz", "epoch600_merged_output.json.gz")

