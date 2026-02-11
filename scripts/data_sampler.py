import numpy as np
import random
import os

DATA_SERIES_DATASET_PATH = "/mnt/hddhelp/mchatzakis/similarity-search-datasets/data_size270M_astronomy_len256_znorm.bin"
DATA_SERIES_QUERIES_PATH = "/mnt/hddhelp/mchatzakis/similarity-search-datasets/queries_ctrl100_astronomy_len256_znorm.bin"

DATASET_NAME = "astronomy"

DIMS = 256
DATASET_SIZE = 270000000
QUERIES_SIZE = 100

DATASET_SAMPLE_SIZE = 50000
QUERIES_SAMPLE_SIZE = 100

# Construct absolute paths from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, '..'))
DATASET_SAMPLE_OUTPUT_PATH = os.path.join(project_root, "data", f"{DATASET_NAME}.data.len{DIMS}.size{DATASET_SAMPLE_SIZE}.znorm.bin")
QUERIES_SAMPLE_OUTPUT_PATH = os.path.join(project_root, "data", f"{DATASET_NAME}.query.len{DIMS}.size{DATASET_SAMPLE_SIZE}.znorm.bin")

np.random.seed(42)
random.seed(42)

def read_data_series_of_file(file_path: str, num_series: int, series_length: int) -> np.ndarray:
    """
    @brief Read a series of data from a binary file.

    @param file_path: Path to the binary file
    @param num_series: Number of series to read
    @param series_length: Length of each series
    @return: A numpy array of shape (num_series, series_length)
    """
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.float32, count=num_series * series_length)
        
    if data.size != num_series * series_length:
        raise ValueError(f"Expected {num_series * series_length} data points, but got {data.size}")
    
    return data.reshape(num_series, series_length)

def save_data_series_to_file(data: np.ndarray, file_path: str) -> None:
    """
    Save a 2D NumPy array to a binary file in float32 format.

    @param data: The NumPy array to save
    @param file_path: Path to the binary output file
    """
    with open(file_path, "wb") as f:
        data.astype(np.float32).tofile(f)

def sample_data(dataset: np.ndarray, sample_size: int) -> np.ndarray:
    """
    Sample a subset of the dataset.

    @param dataset: The dataset to sample from
    @param sample_size: The number of samples to take
    @return: A sampled subset of the dataset
    """
    if sample_size > len(dataset):
        raise ValueError("Sample size cannot be larger than the dataset size.")
    
    indices = random.sample(range(len(dataset)), sample_size)
    return dataset[indices]

def check_data_equality(original: np.ndarray, loaded: np.ndarray, name: str) -> None:
    """
    Compare two datasets and print differences if any are found.

    @param original: The original dataset
    @param loaded: The dataset loaded back from file
    @param name: A name label for printing/logging purposes
    """
    if not np.allclose(original, loaded):
        diff_indices = np.where(~np.isclose(original, loaded))
        num_mismatches = len(diff_indices[0])
        print(f"\n{name} mismatch detected!")
        print(f"Number of mismatches: {num_mismatches}")

        # Show the first 5 mismatches for inspection
        for i in range(min(5, num_mismatches)):
            idx = tuple(d[i] for d in diff_indices)
            orig_value = original[idx]
            loaded_value = loaded[idx]
            print(f"Mismatch at index {idx}: original={orig_value}, loaded={loaded_value}")

        raise AssertionError(f"{name} does not match loaded data. See mismatches above.")
    else:
        print(f"{name} matches successfully.")

dataset_series = read_data_series_of_file(DATA_SERIES_DATASET_PATH, DATASET_SIZE, DIMS)
queries_series = read_data_series_of_file(DATA_SERIES_QUERIES_PATH, QUERIES_SIZE, DIMS)

print("Dataset shape:", dataset_series.shape)
print("Queries shape:", queries_series.shape)

# sample the dataset and queries
dataset_sample = sample_data(dataset_series, DATASET_SAMPLE_SIZE)
queries_sample = sample_data(queries_series, QUERIES_SAMPLE_SIZE)
print("Sampled dataset shape:", dataset_sample.shape)
print("Sampled queries shape:", queries_sample.shape)

# save the sampled dataset and queries to binary files
save_data_series_to_file(dataset_sample, DATASET_SAMPLE_OUTPUT_PATH)
save_data_series_to_file(queries_sample, QUERIES_SAMPLE_OUTPUT_PATH)
print(f"Sampled dataset saved to {DATASET_SAMPLE_OUTPUT_PATH}")
print(f"Sampled queries saved to {QUERIES_SAMPLE_OUTPUT_PATH}")

# Sanity check: load the sampled dataset and queries
loaded_dataset_sample = read_data_series_of_file(DATASET_SAMPLE_OUTPUT_PATH, DATASET_SAMPLE_SIZE, DIMS)
loaded_queries_sample = read_data_series_of_file(QUERIES_SAMPLE_OUTPUT_PATH, QUERIES_SAMPLE_SIZE, DIMS)
print("Loaded sampled dataset shape:", loaded_dataset_sample.shape)
print("Loaded sampled queries shape:", loaded_queries_sample.shape)

# Check if the loaded data matches the original sampled data
#assert np.allclose(dataset_sample, loaded_dataset_sample), f"Loaded dataset sample does not match original, expected {dataset_sample.shape}, got {loaded_dataset_sample.shape}. 10 first elements: {dataset_sample[:10]} vs {loaded_dataset_sample[:10]}"
#assert np.allclose(queries_sample, loaded_queries_sample), f"Loaded queries sample does not match original, expected {queries_sample.shape}, got {loaded_queries_sample.shape}. 10 first elements: {queries_sample[:10]} vs {loaded_queries_sample[:10]}"
check_data_equality(dataset_sample, loaded_dataset_sample, "Dataset sample")
check_data_equality(queries_sample, loaded_queries_sample, "Queries sample")
print("Data sampling and saving completed successfully.")

