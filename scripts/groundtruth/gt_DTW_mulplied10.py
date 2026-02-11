import re
import os

def multiply_file_values(input_filename, output_filename, multiplier):
    # Extract k value from the filename
    match = re.search(r'_k(\d+)\.txt$', input_filename)
    if not match:
        print(f"Error: Could not determine 'k' value from filename: {input_filename}")
        print("Filename must end with '_k<number>.txt'")
        return

    k_value = int(match.group(1))
    print(f"Detected k value: {k_value} from filename: {input_filename}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        with open(input_filename, 'r') as infile:
            lines = infile.readlines()

        with open(output_filename, 'w') as outfile:
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line: # Skip empty lines
                    outfile.write("\n")
                    continue

                str_values = stripped_line.split()
                multiplied_values = []

                for s_val in str_values:
                    try:
                        value = float(s_val)
                        multiplied_values.append(str(value * multiplier))
                    except ValueError:
                        print(f"Warning: Skipping invalid value '{s_val}' in line: '{stripped_line}'")
                        pass 
                    except Exception as e:
                        print(f"An error occurred while processing value '{s_val}': {e}")
                        pass 

                # Join the multiplied values back into a single string, separated by spaces
                outfile.write(" ".join(multiplied_values) + "\n")

        print(f"Successfully multiplied values from '{input_filename}' by {multiplier} and saved to '{output_filename}'.")

    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Define the base path and multiplier (absolute paths from project root)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
base_path = os.path.join(project_root, "tests", "groundtruth", "Distances")
output_base_path = os.path.join(project_root, "tests", "groundtruth", "Distances") # New directory for output

factor = 10

# List of k values to test
k_values = [1, 10, 100]

for k in k_values:
    input_file = os.path.join(base_path, f"raw_bruteForce_gtDTW_D_astronomy_len256_size50000_q100_k{k}.txt")
    output_file = os.path.join(output_base_path, f"bruteForce_gtDTW_D_astronomy_len256_size50000_q100_k{k}.txt")
    print(f"\nProcessing {input_file}...")
    multiply_file_values(input_file, output_file, factor)
