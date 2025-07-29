# **Snake AI Trainer (Double DQN)**

---

### **1. Introduction**
This project implements an AI agent that learns to play the classic Snake game using a Double Deep Q-Network (DDQN) algorithm. The agent is trained to maximize its score by consuming food and avoiding collisions with walls, its own body, and optionally, static obstacles. The project includes both a terminal-based training script and a Graphical User Interface (GUI) for easier control and visualization of the training process.

---

### **2. Features**
* **DDQN Implementation:** Utilizes a Double Deep Q-Network for stable and efficient reinforcement learning.
* **Obstacle Game Mode:** Supports training and playing the Snake game with static obstacles, adding an extra layer of complexity.
* **Training Resumption:** Ability to resume training from the last saved checkpoint, allowing for continuous learning.
* **TensorBoard Integration:** Detailed logging of training metrics (scores, loss, epsilon) to TensorBoard for visual analysis of learning progress.
* **Live Matplotlib Plotting:** Real-time visualization of episode scores and mean scores during training.
* **Tkinter GUI:** A user-friendly graphical interface to start/resume training sessions for both normal and obstacle modes, and to launch TensorBoard.

---

### **3. Technologies Used**
This project leverages the following key technologies:

* **Programming Language:**
    * Python 3.x

* **Core Libraries/Frameworks:**
    * **PyTorch:** For building and training the deep neural networks (Q-Network and Target Q-Network).
    * **Pygame:** Used for rendering the Snake game environment.
    * **NumPy:** For numerical operations and state representation.
    * **TensorBoard:** For visualizing training metrics, graphs, and experiment tracking.
    * **Matplotlib:** For generating real-time plots of scores and mean scores during training.
    * **Tkinter:** Python's standard GUI library, used for the main application interface.

---

### **4. Getting Started**
Follow these instructions to set up and run the project on your local machine.

#### **4.1. Prerequisites**
Ensure you have the following software installed:

* **Python 3.x:** Download from [python.org](https://www.python.org/). It's recommended to use Python 3.8 or newer.
* **Git:** Download from [git-scm.com](https://git-scm.com/downloads).

#### **4.2. Installation**

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
    cd [your-repo-name]
    ```
    (Replace `[your-username]` and `[your-repo-name]` with your actual GitHub details)

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    * **Note on Pygame/Torch/TensorBoard:** If you encounter installation issues, especially with `pygame` or `torch`, ensure your Python version is compatible. For `torch`, consider installing it from the official PyTorch website (pytorch.org) for specific CUDA versions if you have an NVIDIA GPU, then `pip install -r requirements.txt` again for other dependencies.
    * **OpenCV (`cv2`)**: While `cv2` is not directly imported in the provided files, if you had previous issues with it, the most common way to install is `pip install opencv-python`. If that fails, `pip install opencv-python-headless` is often a solution for server environments or when GUI components are not needed.

---

### **5. Usage**

You can run the AI training and game through the provided GUI or directly via the terminal.

#### **5.1. Using the GUI (Recommended)**

1.  **Activate your virtual environment** (if not already active):
    * **Windows:** `.\venv\Scripts\activate`
    * **macOS/Linux:** `source venv/bin/activate`

2.  **Navigate to the `GUI` directory:**
    ```bash
    cd GUI
    ```

3.  **Run the GUI application:**
    ```bash
    python main_gui.py
    ```
    The GUI window will appear, allowing you to:
    * **Start New Normal Game AI:** Begins training a new agent without obstacles.
    * **Resume Normal Game AI:** Resumes training a normal game agent from the latest checkpoint.
    * **Start New Obstacle Game AI:** Begins training a new agent with static obstacles.
    * **Resume Obstacle Game AI:** Resumes training an obstacle game agent from the latest checkpoint.
    * **Launch TensorBoard:** Opens TensorBoard in your default web browser to visualize training progress.

#### **5.2. Using the Terminal (For Direct Training)**

1.  **Activate your virtual environment** (if not already active):
    * **Windows:** `.\venv\Scripts\activate`
    * **macOS/Linux:** `source venv/bin/activate`

2.  **Navigate to the `terminal` directory:**
    ```bash
    cd terminal
    ```

3.  **Run the training script:**
    * **Start new normal training:**
        ```bash
        python train.py
        ```
    * **Start new training with obstacles:**
        ```bash
        python train.py --obstacles
        ```
    * **Resume normal training:**
        ```bash
        python train.py --resume
        ```
    * **Resume training with obstacles:**
        ```bash
        python train.py --obstacles --resume
        ```
    * **Specify a random seed (e.g., for reproducibility):**
        ```bash
        python train.py --seed 123
        ```
    * You will see training output in your terminal, and a Matplotlib plot will appear and update live. TensorBoard logs will be generated in the `runs/` directory at the root of your project.

#### **5.3. Launching TensorBoard Separately**

If you launched training via the terminal and want to view logs, or if you prefer to launch TensorBoard manually:

1.  **Activate your virtual environment.**
2.  **Navigate to the root of your project directory** (`snake-game_ai_double-Q/`).
3.  **Run TensorBoard:**
    ```bash
    tensorboard --logdir runs
    ```
    TensorBoard will then tell you the local URL (usually `http://localhost:6006/`) to open in your web browser.

---

### **6. Screenshots/Demos**
*(Replace these with actual links to your images or GIFs)*

* **GUI Main Window:** A visual representation of the application's starting interface.
    ![GUI Main Window](https://via.placeholder.com/700x400?text=Snake+AI+Trainer+GUI)

* **Snake Game in Action:** A screenshot or GIF of the AI playing the Snake game (normal mode).
    ![Snake Game Normal Mode](https://via.placeholder.com/700x400?text=Snake+Game+Normal+Mode+AI+Playing)

* **Snake Game with Obstacles:** A screenshot or GIF of the AI playing the Snake game with obstacles.
    ![Snake Game Obstacles Mode](https://via.placeholder.com/700x400?text=Snake+Game+Obstacles+Mode+AI+Playing)

* **TensorBoard Metrics Dashboard:** An example of the training curves (scores, loss, epsilon) in TensorBoard.
    ![TensorBoard Dashboard](https://via.placeholder.com/700x400?text=TensorBoard+Training+Metrics)

* **Live Matplotlib Plot:** A screenshot of the live score plot generated during terminal training.
    ![Matplotlib Live Plot](https://via.placeholder.com/700x400?text=Matplotlib+Live+Score+Plot)

---

### **7. Project Structure**

### **8. Contributing**
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

### **9. License**
This project is licensed under the **MIT License** - see the `LICENSE` file for details. (Create a `LICENSE` file in the root of your project if you don't have one).

---

### **10. Contact**
[Your Name] - [your.email@example.com]
Project Link: [https://github.com/your-username/your-repo-name](https://github.com/your-username/your-repo-name)

---
