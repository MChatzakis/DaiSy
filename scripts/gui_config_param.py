import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import platform
from typing import Optional, Tuple

from diNoSimilaritySearch import BruteForceSearch, LbBruteforceSearch #, MessiSearch, OdysseySearch, ParISSearch, SingSearch

# Global variables
user_inputs = {}

dist_options = [("Squared Euclidean (L2²)", "L2_SQUARED")]

search_classes = {
    "Brute Force": BruteForceSearch,
    "Lower Bound Brute Force": LbBruteforceSearch,
    # "Messi": MessiSearch,
    # "Odyssey": OdysseySearch,
    # "ParIS": ParISSearch,
    # "Sing": SingSearch,
}

search_options = list(search_classes.keys())

thread_options = [1, 4, 8]

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

    if not user_dataset_path.get() or not user_query_path.get():
        messagebox.showerror("Input Error", "Please select both dataset and query files.")
        return

    try:
        ndb_val = int(ndb_entry.get())
        n_query_val = int(n_query_entry.get())
        dim_val = int(dim_entry.get())
        query_val = int(query_entry.get())
        k_val = int(k_entry.get())
        threads_val = int(threads_var.get())
        
        # Validate non-negative integers
        if any(val < 0 for val in [ndb_val, n_query_val, dim_val, query_val, k_val, threads_val]):
            messagebox.showerror("Input Error", "All numeric values must be non-negative.")
            return

        # query_val > n_query_val 
        if query_val > n_query_val:
            messagebox.showerror("Input Error", "Query Number cannot exceed Number of Query Vectors.")
            return

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid integers for all numeric fields.")
        return

    user_inputs = {
        "Dataset Path": user_dataset_path.get(),
        "Query Path": user_query_path.get(),
        "Number of Database Vectors": ndb_val,
        "Number of Query Vectors": n_query_val, 
        "Vector Dimensionality": dim_val,
        "Query Number": query_val,
        "k-Nearest Neighbors": k_val,
        "Distance Metric": distance_var.get(),
        "Search Method": search_var.get(),
        "Threads": threads_val,
    }

    messagebox.showinfo("Configuration Saved", f"The following parameters have been saved:\n{user_inputs}")
    print("Stored inputs:", user_inputs)
    window.quit() 

def get_config() -> dict:
    """
    @brief Launch the GUI window and return user-selected similarity search configuration.

    @return dict: User input parameters collected from the GUI
    """    
    global user_inputs
    user_inputs = {}
    window.mainloop()
    return user_inputs

def param_gui() -> Optional[Tuple[str, str, int, int, int, int, int, str, str, int]]:
    """
    @brief Launch GUI and collect user-defined parameters for search.

    @return: Tuple,
             or None if no configuration is provided
    @throws KeyError: If a required configuration key is missing
    """    
    config = get_config()
    if config:
        try:
            return (
                config["Dataset Path"],
                config["Query Path"],
                config["Number of Database Vectors"],   
                config["Number of Query Vectors"],   
                config["Vector Dimensionality"],        
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


# ====== GUI Setup ======
window = tk.Tk()
window.title("Similarity Search Configuration")
window.configure(bg="#f0f0f5")

window_width = 700
window_height = 900
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

# ===== Scrollable Main Frame Setup =====
container = ttk.Frame(window)
container.pack(fill="both", expand=True, padx=20, pady=10)

canvas = tk.Canvas(container, borderwidth=0, background="#f0f0f5", highlightthickness=0)
scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

def _on_mousewheel(event):
    if platform.system() == 'Windows':
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    elif platform.system() == 'Darwin':  # macOS
        canvas.yview_scroll(int(-1 * (event.delta)), "units")
    else:  # Linux
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")

def _bind_mousewheel(event):
    if platform.system() in ('Windows', 'Darwin'):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    else:
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

def _unbind_mousewheel(event):
    if platform.system() in ('Windows', 'Darwin'):
        canvas.unbind_all("<MouseWheel>")
    else:
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")
        
scrollable_frame.bind("<Enter>", _bind_mousewheel)
scrollable_frame.bind("<Leave>", _unbind_mousewheel)

# Use scrollable_frame as your main container now:
main_frame = scrollable_frame

# Dataset File Selection Frame
dataset_frame = ttk.LabelFrame(main_frame, text="Data Files", padding=10)
dataset_frame.pack(fill="x", pady=10)

user_dataset_path = tk.StringVar()
user_query_path = tk.StringVar()

def select_user_dataset():
    path = filedialog.askopenfilename(title="Select Dataset File", filetypes=[("Binary files", "*.bin"), ("All files", "*.*")])
    if path:
        user_dataset_path.set(path)

def select_user_queries():
    path = filedialog.askopenfilename(title="Select Query File", filetypes=[("Binary files", "*.bin"), ("All files", "*.*")])
    if path:
        user_query_path.set(path)

ttk.Button(dataset_frame, text="Load Dataset File", command=select_user_dataset).pack(anchor="w", pady=2)
ttk.Entry(dataset_frame, textvariable=user_dataset_path, state="readonly", width=80).pack(anchor="w", pady=(0, 5))

ttk.Button(dataset_frame, text="Load Query File", command=select_user_queries).pack(anchor="w", pady=2)
ttk.Entry(dataset_frame, textvariable=user_query_path, state="readonly", width=80).pack(anchor="w", pady=(0, 5))

# Parameters Frame
input_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
input_frame.pack(fill="x", pady=10)

# Number of Database Vectors (n_database)
ndb_label = ttk.Label(input_frame, text="Number of Database Vectors (n_database):")
ndb_label.pack(anchor="w", pady=(10, 0))
ndb_entry = ttk.Entry(input_frame, width=13)
ndb_entry.pack(anchor="w")
CreateToolTip(ndb_entry, "Specify the number of database vectors.")

# Number of Query Vectors (n_database)
nquery_label = ttk.Label(input_frame, text="Number of Query Vectors (n_query):")
nquery_label.pack(anchor="w", pady=(10, 0))
n_query_entry = ttk.Entry(input_frame, width=13)
n_query_entry.pack(anchor="w")
CreateToolTip(n_query_entry, "Specify the number of query vectors.")

# Vector Dimensionality (dim)
dim_label = ttk.Label(input_frame, text="Vector Dimensionality (dim):")
dim_label.pack(anchor="w", pady=(10, 0))
dim_entry = ttk.Entry(input_frame, width=13)
dim_entry.pack(anchor="w")
CreateToolTip(dim_entry, "Specify the dimensionality of vectors.")

# Number of Queries
query_label = ttk.Label(input_frame, text="Number of Queries (n_query):")
query_label.pack(anchor="w")
query_entry = ttk.Entry(input_frame, width=13)
query_entry.pack(anchor="w")
CreateToolTip(query_entry, "Specify how many queries should be processed.")

# K-Nearest Neighbors 
k_label = ttk.Label(input_frame, text="Number of Nearest Neighbors (k):")
k_label.pack(anchor="w", pady=(10, 0))
k_entry = ttk.Entry(input_frame, width=13)
k_entry.pack(anchor="w")
CreateToolTip(k_entry, "Enter the number of closest neighbors to retrieve.")

# Distance Metric
distance_var = tk.StringVar(value="L2_SQUARED")
distance_frame = ttk.LabelFrame(main_frame, text="Distance Metric", padding=10)
distance_frame.pack(fill="x", pady=10)
for text, val in dist_options:
    ttk.Radiobutton(distance_frame, text=text, variable=distance_var, value=val).pack(anchor="w", pady=2)

# Search Method
search_var = tk.StringVar(value=search_options[0])
search_frame = ttk.LabelFrame(main_frame, text="Search Method", padding=10)
search_frame.pack(fill="x", pady=10)
for option in search_options:
    ttk.Radiobutton(search_frame, text=option, variable=search_var, value=option).pack(anchor="w", pady=2)

# Threads
threads_var = tk.IntVar(value=1)
threads_frame = ttk.LabelFrame(main_frame, text="Number of Threads", padding=10)
threads_frame.pack(fill="x", pady=10)
ttk.Label(threads_frame, text="Select number of threads to use:").pack(side="left", padx=(0, 10))
threads_combo = ttk.Combobox(threads_frame, textvariable=threads_var, values=thread_options, state="readonly", width=5)
threads_combo.pack(side="left")
CreateToolTip(threads_combo, "Choose how many threads the search should utilize.")

# Buttons
btn_frame = tk.Frame(window, bg="#f0f0f5", pady=20)
btn_frame.pack()

tk.Button(btn_frame, text="Confirm Configuration", bg="#4CAF50", fg="white",
          activebackground="#45a049", command=store_input).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="Cancel", bg="#d3312b", fg="white",
          activebackground="#c62828", command=window.quit).pack(side=tk.LEFT, padx=10)