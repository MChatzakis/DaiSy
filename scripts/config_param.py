import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Tuple
import sys
import os
import re

from diNoSimilaritySearch import BruteForceSearch, DistanceType #, LbBruteForceSearch, Messi, Odyssey, Paris, Sing

# Global variables
user_inputs = {}

dataset_options = ["Random", "Astronomy"]

dataset_limits = {
    "Random": {"queries": (1, 1000), "k": (1, 1000)},
    "Astronomy": {"queries": (1, 50000), "k": (1, 50000)},
}

dist_options = [("Squared Euclidean (L2²)", "L2_SQUARED")]

search_classes = {
    "Brute Force": BruteForceSearch,
    # "Lower Bound Brute Force": LbBruteForceSearch,
    # "Messi": Messi,
    # "Odyssey": Odyssey,
    # "Paris": Paris,
    # "Sing": Sing,
}

search_options = list(search_classes.keys())

thread_options = [1, 4, 8]

def lowercaseFirstLetter(s: str) -> str:
    """
    @brief Convert the first character of the string to lowercase.
    
    @param s : Input string
    @return str: String with the first character lowercased
    """    
    if not s:
        return s  # Return empty string 
    return s[0].lower() + s[1:]

class CreateToolTip:
    """
    @brief Helper class to create a tooltip for a tkinter widget.

    @param widget: The tkinter widget to attach the tooltip to
    @param text: The tooltip text to display
    """

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tooltip = None
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)

    def enter(self, event: tk.Event | None = None) -> None:
        """
        @brief Display the tooltip near the widget when mouse enters.
        """        
        if self.tooltip:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tooltip, text=self.text, justify="left",
            background="lightyellow", relief="solid", borderwidth=1,
            font=("tahoma", 10, "normal")
        )
        label.pack(ipadx=1)

    def leave(self, event: tk.Event | None = None) -> None:
        """
        @brief Hide and destroy the tooltip when mouse leaves the widget.
        """        
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

def store_input() -> None:
    """
    @brief Validate and store user input from the GUI form.

    @return: None (closes the GUI window after saving inputs)
    """    
    global user_inputs
    try:
        query_number = int(query_entry.get())
        k_neighbors = int(k_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid integers for the query count and number of neighbors.")
        return

    dataset = dataset_var.get()
    q_min, q_max = dataset_limits[dataset]["queries"]
    k_min, k_max = dataset_limits[dataset]["k"]

    if not (q_min <= query_number <= q_max):
        messagebox.showerror("Input Error", f"Number of Queries must be between {q_min} and {q_max}.")
        return
    if not (k_min <= k_neighbors <= k_max):
        messagebox.showerror("Input Error", f"Number of Nearest Neighbors (k) must be between {k_min} and {k_max}.")
        return

    user_inputs = {
        "Dataset": lowercaseFirstLetter(dataset),
        "Query Number": query_number,
        "k-Nearest Neighbors": k_neighbors,
        "Distance Metric": distance_var.get(),
        "Search Method": search_var.get(),
        "Threads": threads_var.get()
    }

    messagebox.showinfo("Configuration Saved", f"The following parameters have been saved:\n{user_inputs}")
    print("Stored inputs:", user_inputs)
    window.quit()  


def validate_int_range(new_value: str, min_val: int, max_val: int) -> bool:
    """
    @brief Validate that a string represents an integer within a specified range.

    @param new_value: The new string value from entry widget
    @param min_val: Minimum allowed integer value
    @param max_val: Maximum allowed integer value
    @return bool: True if new_value is empty or an integer within [min_val, max_val], 
                    False otherwise
    """    
    if new_value == "":
        return True 
    try:
        val = int(new_value)
        return min_val <= val <= max_val
    except ValueError:
        return False

def update_limits(*args) -> None:
    """
    @brief Update input validation and tooltips based on the currently selected dataset.

    @return: None
    """    
    dataset = dataset_var.get()
    q_min, q_max = dataset_limits[dataset]["queries"]
    k_min, k_max = dataset_limits[dataset]["k"]
 
    CreateToolTip(query_entry, f"Specify queries between {q_min} and {q_max}.")
    CreateToolTip(k_entry, f"Specify k between {k_min} and {k_max}.")
 
    query_vcmd = (window.register(lambda val: validate_int_range(val, q_min, q_max)), "%P")
    k_vcmd = (window.register(lambda val: validate_int_range(val, k_min, k_max)), "%P")

    query_entry.config(validate="key", validatecommand=query_vcmd)
    k_entry.config(validate="key", validatecommand=k_vcmd)

def get_config() -> dict:
    """
    @brief Launch the GUI window and return user-selected similarity search configuration.

    @return dict: User input parameters collected from the GUI
    """    
    global user_inputs
    user_inputs = {}

    window.mainloop()

    return user_inputs

def param_gui() -> Optional[Tuple[str, int, int, str, str, int]]:
    """
    @brief Launch GUI and collect user-defined parameters for search.

    @return: Tuple containing (Dataset name, Query Number, kNN, Distance Metric, Search Method, Threads),
             or None if no configuration is provided
    @throws KeyError: If a required configuration key is missing
    """    
    config = get_config()  # This will open the GUI
    
    if config:
        try:
            return (
                config["Dataset"],
                config["Query Number"],
                config["k-Nearest Neighbors"],
                config["Distance Metric"],
                config["Search Method"],
                config["Threads"]
            )
        except KeyError as e:
            print(f"Configuration key missing: {e}")
    else:
        print("No configuration was provided.")
    
    return None

def find_data_files(db_name: str, data_folder: str = '../data') -> Tuple[str, str, int, int]:
    """
    @brief Locate the dataset and query files for the given database name.

    @param db_name: Base name of the dataset
    @param data_folder: Directory where data files are stored
    @return: Tuple (dataset_path, query_path, dim, nb) where:
             - dataset_path: path to the dataset file
             - query_path: path to the query file
             - dim: vector dimensionality
             - nb: number of database vectors
    @throws FileNotFoundError: If matching dataset or query files are not found
    """    
    pattern_db = re.compile(
        rf"{re.escape(db_name)}\.data(?:\.[^.]+)?\.len(\d+)\.size(\d+)\.znorm\.bin"
    )

    pattern_query = re.compile(
        rf"{re.escape(db_name)}\.query(?:\.[^.]+)?\.len(\d+)\.size(\d+)(?:\.znorm)?\.bin"
    )

    dataset_path = None
    query_path = None
    dim = None
    nb = None

    for fname in os.listdir(data_folder):
        # Search for dataset file
        m_db = pattern_db.match(fname)
        if m_db:
            dataset_path = os.path.join(data_folder, fname)
            dim = int(m_db.group(1))
            nb = int(m_db.group(2))

        # Search for query file
        m_query = pattern_query.match(fname)
        if m_query:
            query_path = os.path.join(data_folder, fname)

    if dataset_path is None or query_path is None:
        raise FileNotFoundError(f"Could not find dataset or query files for '{db_name}' in '{data_folder}'")

    return dataset_path, query_path, dim, nb

# ====== GUI Setup ======
window = tk.Tk()
window.title("Similarity Search Configuration")
window.configure(bg="#f0f0f5")

window_width = 700
window_height = 850
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Style
style = ttk.Style(window)
style.configure("TLabel", font=("Arial", 11))
style.configure("TButton", font=("Arial", 11, "bold"), padding=6)
style.configure("Accent.TButton", background="#4CAF50", foreground="white")
style.map("Accent.TButton", background=[("active", "#45a049")])

# Title
title = ttk.Label(window, text="Configure Similarity Search Parameters",
                  font=("Helvetica", 16, "bold"), background="#f0f0f5")
title.pack(pady=20)

# Main Frame
main_frame = ttk.Frame(window, padding=20)
main_frame.pack(fill="both", expand=True, padx=30, pady=10)

# Dataset Frame with Queries and k inside
dataset_var = tk.StringVar(value="Random")
dataset_frame = ttk.LabelFrame(main_frame, text="Dataset", padding=10)
dataset_frame.pack(fill="x", pady=10)

# Dataset Radio Buttons
for option in dataset_options:
    ttk.Radiobutton(dataset_frame, text=option, variable=dataset_var, value=option).pack(anchor="w", pady=2)

# Frame inside Dataset Frame for queries and k (side by side)
input_frame = ttk.Frame(dataset_frame)
input_frame.pack(fill="x", pady=(10, 0))

# Number of Queries
query_label = ttk.Label(input_frame, text="Number of Queries:", width=0)
query_label.pack(side="left", padx=(0, 5))
query_entry = ttk.Entry(input_frame, width=3)
query_entry.pack(side="left")
CreateToolTip(query_entry, "Specify how many queries should be processed.")

# Spacer
ttk.Label(input_frame, text="       ").pack(side="left")  

# k Nearest Neighbors
k_label = ttk.Label(input_frame, text="Number of Nearest Neighbors (k):", width=0)
k_label.pack(side="left", padx=(10, 5))
k_entry = ttk.Entry(input_frame, width=3)
k_entry.pack(side="left")
CreateToolTip(k_entry, "Enter the number of closest neighbors to retrieve.")

dataset_var.trace_add("write", update_limits)
update_limits()

# Distance Metric
distance_var = tk.StringVar(value="L2")
distance_frame = ttk.LabelFrame(main_frame, text="Distance Metric", padding=10)
distance_frame.pack(fill="x", pady=10)
for text, val in dist_options:
    ttk.Radiobutton(distance_frame, text=text, variable=distance_var, value=val).pack(anchor="w", pady=2)

# Search Method
search_var = tk.StringVar(value="Brute Force Search")
search_frame = ttk.LabelFrame(main_frame, text="Search Method", padding=10)
search_frame.pack(fill="x", pady=10)
for option in search_options:
    ttk.Radiobutton(search_frame, text=option, variable=search_var, value=option).pack(anchor="w", pady=2)

# Number of Threads
threads_var = tk.IntVar(value=1)
threads_frame = ttk.LabelFrame(main_frame, text="Number of Threads", padding=10)
threads_frame.pack(fill="x", pady=10)

threads_label = ttk.Label(threads_frame, text="Select number of threads to use:")
threads_label.pack(side="left", padx=(0, 10))

threads_combo = ttk.Combobox(threads_frame, textvariable=threads_var, values=thread_options, state="readonly", width=5)
threads_combo.pack(side="left")
CreateToolTip(threads_combo, "Choose how many threads the search should utilize.")

# Action Buttons (with custom background preserved)
btn_frame = tk.Frame(window, bg="#f0f0f5", pady=20)
btn_frame.pack()

store_button = tk.Button(
    btn_frame, text="Confirm Configuration", bg="#4CAF50", fg="white", activebackground="#45a049", command=store_input
)
store_button.pack(side=tk.LEFT, padx=10)

quit_button = tk.Button(
    btn_frame, text="Cancel", bg="#d3312b", fg="white", activebackground="#c62828", command=window.quit
)
quit_button.pack(side=tk.LEFT, padx=10)
