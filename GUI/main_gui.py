# GUI/main_gui.py
import tkinter as tk
from tkinter import ttk # Import ttk for themed widgets
from tkinter import messagebox
import threading
import subprocess
import os
import sys
# Removed matplotlib.pyplot as plt import

# Import the train function from your train.py in the same folder
from train import train
# Import LOG_DIR from config.py in the same folder
from config import LOG_DIR 
# Removed from Helper import plot

class SnakeAIGUI:
    def __init__(self, master):
        self.master = master
        master.title("Snake AI Trainer (DDQN GUI)")
        master.geometry("450x580") # Increased window height again to ensure all buttons are visible and add more space
        master.resizable(False, False)

        # Apply a ttk theme for a more modern look
        self.style = ttk.Style()
        self.style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'

        # Configure some styles
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'), padding=8)
        self.style.map('TButton',
                       background=[('active', '#e0e0e0'), ('!disabled', '#d0d0d0')],
                       foreground=[('active', 'black'), ('!disabled', 'black')])
        self.style.configure('TLabelFrame', background='#f0f0f0', font=('Arial', 12, 'bold'))
        self.style.configure('TEntry', padding=5)


        self.training_thread = None
        # Removed self.plot_figure and self.plot_axes

        # --- Main Frame ---
        self.main_frame = ttk.Frame(master, padding="20 20 20 20")
        self.main_frame.pack(fill="both", expand=True)

        # --- Title ---
        self.title_label = ttk.Label(self.main_frame, text="Snake AI Training Control", font=("Arial", 18, "bold"), foreground="#333")
        self.title_label.pack(pady=15)

        # --- Seed Input ---
        self.seed_frame = ttk.Frame(self.main_frame)
        self.seed_frame.pack(pady=10)
        self.seed_label = ttk.Label(self.seed_frame, text="Random Seed (e.g., 42):")
        self.seed_label.pack(side="left", padx=5)
        self.seed_var = tk.StringVar(master, value="42") # Default seed
        self.seed_entry = ttk.Entry(self.seed_frame, textvariable=self.seed_var, width=15, justify='center')
        self.seed_entry.pack(side="left", padx=5)

        # --- Normal Game Buttons ---
        self.normal_frame = ttk.LabelFrame(self.main_frame, text="Normal Game Mode (DDQN)", padding="10 10 10 10")
        self.normal_frame.pack(pady=10, padx=20, fill="x")

        self.start_normal_button = ttk.Button(self.normal_frame, text="Start New Normal Game AI", command=self.start_normal_game)
        self.start_normal_button.pack(fill="x", pady=5)

        self.resume_normal_button = ttk.Button(self.normal_frame, text="Resume Normal Game AI", command=self.resume_normal_game)
        self.resume_normal_button.pack(fill="x", pady=5)

        # --- Obstacle Game Buttons ---
        self.obstacle_frame = ttk.LabelFrame(self.main_frame, text="Obstacle Game Mode (DDQN)", padding="10 10 10 10")
        self.obstacle_frame.pack(pady=10, padx=20, fill="x")

        self.start_obstacle_button = ttk.Button(self.obstacle_frame, text="Start New Obstacle Game AI", command=self.start_obstacle_game)
        self.start_obstacle_button.pack(fill="x", pady=5)

        self.resume_obstacle_button = ttk.Button(self.obstacle_frame, text="Resume Obstacle Game AI", command=self.resume_obstacle_game)
        self.resume_obstacle_button.pack(fill="x", pady=5) # This button should now be visible

        # --- Visualization Buttons Frame ---
        self.viz_frame = ttk.Frame(self.main_frame)
        self.viz_frame.pack(pady=20) # Increased pady for more space above TensorBoard button

        self.tensorboard_button = ttk.Button(self.viz_frame, text="Launch TensorBoard", command=self.launch_tensorboard, style='Accent.TButton')
        self.tensorboard_button.pack(padx=10) # No side="left" needed if only one button

        # Removed Matplotlib button

        # Custom style for accent button
        self.style.configure('Accent.TButton', background='#4CAF50', foreground='white', font=('Arial', 10, 'bold'))
        self.style.map('Accent.TButton', background=[('active', '#66BB6A')])


        # --- Status Label ---
        self.status_label = ttk.Label(self.main_frame, text="Ready.", foreground="green", font=('Arial', 10, 'italic'))
        self.status_label.pack(pady=10) # Increased pady for more space below status label

    def _get_seed(self):
        try:
            return int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Invalid Seed", "Please enter a valid integer for the random seed.")
            return None

    def _start_training_in_thread(self, enable_obstacles: bool, resume_training: bool):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Training Running", "A training session is already active. Please wait for it to finish or close the Pygame window.")
            return

        seed = self._get_seed()
        if seed is None:
            return

        mode_text = "Obstacles" if enable_obstacles else "Normal"
        action_text = "Resuming" if resume_training else "Starting New"
        self.status_label.config(text=f"{action_text} {mode_text} Game AI...", foreground="blue")
        
        # Create a new thread to run the training
        # Target train function is from the local train.py
        self.training_thread = threading.Thread(target=train, args=(enable_obstacles, resume_training, seed))
        self.training_thread.daemon = True # Allow the main program to exit even if thread is running
        self.training_thread.start()
        
        messagebox.showinfo("Training Started", f"{action_text} {mode_text} Snake AI training. Check the terminal for detailed output and TensorBoard for plots.")

    def start_normal_game(self):
        self._start_training_in_thread(enable_obstacles=False, resume_training=False)

    def resume_normal_game(self):
        self._start_training_in_thread(enable_obstacles=False, resume_training=True)

    def start_obstacle_game(self):
        self._start_training_in_thread(enable_obstacles=True, resume_training=False)

    def resume_obstacle_game(self):
        self._start_training_in_thread(enable_obstacles=True, resume_training=True)

    def launch_tensorboard(self):
        try:
            if sys.platform == "win32":
                tensorboard_exe = os.path.join(sys.prefix, 'Scripts', 'tensorboard.exe')
            else: # Linux/macOS
                tensorboard_exe = os.path.join(sys.prefix, 'bin', 'tensorboard')

            if not os.path.exists(tensorboard_exe):
                messagebox.showerror("Error", f"TensorBoard executable not found at {tensorboard_exe}. Ensure it's installed in your virtual environment and the environment is active.")
                return

            self.status_label.config(text="Launching TensorBoard...", foreground="purple")
            
            # Change directory to the GUI folder for TensorBoard to find 'runs' correctly
            current_dir = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(__file__))) # Change to current script's directory

            subprocess.Popen([tensorboard_exe, "--logdir", LOG_DIR])
            
            os.chdir(current_dir) # Change back to the original directory after launching TensorBoard

            messagebox.showinfo("TensorBoard", f"TensorBoard launched. Open your browser to http://localhost:6006/ (or the address shown in the terminal where TensorBoard was launched).")
            self.status_label.config(text="TensorBoard launched.", foreground="green")

        except FileNotFoundError:
            messagebox.showerror("Error", "TensorBoard command not found. Please ensure TensorBoard is installed in your virtual environment and your environment is active.")
            self.status_label.config(text="Failed to launch TensorBoard.", foreground="red")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while launching TensorBoard: {e}")
            self.status_label.config(text="Failed to launch TensorBoard.", foreground="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = SnakeAIGUI(root)
    root.mainloop()
