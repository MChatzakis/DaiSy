import tkinter as tk
from tkinter import ttk, messagebox

# Global variables
user_inputs = {}

dataset_options = ["Random", "Astronomy"]

dataset_limits = {
    "Random": {"queries": (1, 1000), "k": (1, 1000)},
    "Astronomy": {"queries": (1, 50000), "k": (1, 50000)},
}

dist_options = [("Squared Euclidean (L2²)", "L2_SQUARED")]

search_options = [
    "BruteForceSearch",
    "LbBruteForceSearch",
    "Messi",
    "Odyssey",
    "Paris",
    "Sing",
]

thread_options = [1, 4, 8]

def lowercaseFirstLetter(s: str) -> str:
    if not s:
        return s  # Return empty string 
    return s[0].lower() + s[1:]

# Tooltip Helper
class CreateToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
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

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# Store Input Logic 
def store_input():
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

# Main Window
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

# Validation function for entries
def validate_int_range(new_value, min_val, max_val):
    if new_value == "":
        return True 
    try:
        val = int(new_value)
        return min_val <= val <= max_val
    except ValueError:
        return False

# Function to update validation and tooltips based on selected dataset
def update_limits(*args):
    dataset = dataset_var.get()
    q_min, q_max = dataset_limits[dataset]["queries"]
    k_min, k_max = dataset_limits[dataset]["k"]
 
    CreateToolTip(query_entry, f"Specify queries between {q_min} and {q_max}.")
    CreateToolTip(k_entry, f"Specify k between {k_min} and {k_max}.")
 
    query_vcmd = (window.register(lambda val: validate_int_range(val, q_min, q_max)), "%P")
    k_vcmd = (window.register(lambda val: validate_int_range(val, k_min, k_max)), "%P")

    query_entry.config(validate="key", validatecommand=query_vcmd)
    k_entry.config(validate="key", validatecommand=k_vcmd)

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

# window.mainloop()

def get_config():
    global user_inputs
    user_inputs = {}

    window.mainloop()

    return user_inputs
