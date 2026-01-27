import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import time
import threading

# --------------------------------------------------------
# GridWorld Environment with Key System
# --------------------------------------------------------
class GridWorld:
    def __init__(self):
        self.size = 6  # Increased grid size
        self.start = (0, 0)
        self.goal = (5, 5)
        self.key_pos = (2, 4)  # Key position (critical reward)
        self.pit1 = (2, 2)
        self.pit2 = (4, 1)
        self.obstacles = [(1, 3), (3, 1), (1, 4), (4, 3), (3, 4), (5, 2)]  # More obstacles
        self.state = self.start
        self.has_key = False
        
    def reset(self):
        self.state = self.start
        self.has_key = False
        return self.get_full_state()
    
    def get_full_state(self):
        # State includes position and key status
        return (*self.state, int(self.has_key))
    
    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.size - 1, y + 1)
        
        # Check if new position is an obstacle
        if (x, y) in self.obstacles:
            x, y = self.state  # Stay in place
            
        self.state = (x, y)
        
        # Check for key pickup (CRITICAL SECTION)
        if self.state == self.key_pos and not self.has_key:
            self.has_key = True
            return self.get_full_state(), 15, False  # Big reward for key!
        
        # Check for goal (need key to win)
        if self.state == self.goal:
            if self.has_key:
                return self.get_full_state(), 50, True  # Win with key!
            else:
                return self.get_full_state(), -5, False  # Penalty for reaching goal without key
        
        # Check for pits
        if self.state in [self.pit1, self.pit2]:
            return self.get_full_state(), -20, True
        
        return self.get_full_state(), -1, False

# --------------------------------------------------------
# RL Agent — Q Learning with Key State
# --------------------------------------------------------
class Agent:
    def __init__(self, env):
        self.env = env
        self.actions = [0, 1, 2, 3]
        # Q-table now includes key state: (x, y, has_key, action)
        self.q = np.zeros((6, 6, 2, 4))
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_decay = 0.997
        self.epsilon_min = 0.01
        
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        x, y, has_key = state
        return np.argmax(self.q[x, y, has_key])
    
    def learn(self, state, action, reward, next_state):
        x, y, has_key = state
        nx, ny, next_has_key = next_state
        old = self.q[x, y, has_key, action]
        new = reward + self.gamma * np.max(self.q[nx, ny, next_has_key])
        self.q[x, y, has_key, action] += self.lr * (new - old)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --------------------------------------------------------
# Tkinter GUI
# --------------------------------------------------------
class GameGUI:
    def __init__(self):
        self.env = GridWorld()
        self.agent = Agent(self.env)
        
        self.total_episodes = 0
        self.total_rewards = []
        self.is_training = False
        
        # ---------------- MAIN WINDOW ----------------
        self.window = tk.Tk()
        self.window.title("Q-Learning GridWorld with Key System")
        self.window.configure(bg="#0d1117")
        self.window.resizable(False, False)
        
        # Main container
        main_frame = tk.Frame(self.window, bg="#0d1117")
        main_frame.pack(padx=20, pady=20)
        
        # Left panel - Grid
        self.frame_grid = tk.Frame(main_frame, bg="#161b22", relief="solid", borderwidth=2)
        self.frame_grid.grid(row=0, column=0, padx=10, pady=10)
        
        # Title for grid
        grid_title = tk.Label(
            self.frame_grid, 
            text="🎮 GridWorld with Key System",
            bg="#161b22", 
            fg="#58a6ff",
            font=("Arial", 16, "bold")
        )
        grid_title.pack(pady=10)
        
        # Canvas for grid
        self.canvas = tk.Canvas(
            self.frame_grid, 
            width=600, 
            height=600, 
            bg="#0d1117",
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Right panel - Controls & Stats
        self.frame_controls = tk.Frame(main_frame, bg="#161b22", relief="solid", borderwidth=2)
        self.frame_controls.grid(row=0, column=1, sticky="ns", padx=10, pady=10)
        
        # Title
        control_title = tk.Label(
            self.frame_controls,
            text="🤖 Control Panel",
            bg="#161b22",
            fg="#58a6ff",
            font=("Arial", 16, "bold")
        )
        control_title.pack(pady=15)
        
        # Stats Frame
        stats_frame = tk.Frame(self.frame_controls, bg="#0d1117", relief="solid", borderwidth=1)
        stats_frame.pack(pady=10, padx=15, fill="x")
        
        tk.Label(
            stats_frame,
            text="📊 Statistics",
            bg="#0d1117",
            fg="#79c0ff",
            font=("Arial", 12, "bold")
        ).pack(pady=5)
        
        # Episode counter
        self.episode_label = tk.Label(
            stats_frame,
            text="Episodes: 0",
            bg="#0d1117",
            fg="#8b949e",
            font=("Arial", 11)
        )
        self.episode_label.pack(pady=3)
        
        # Epsilon value
        self.epsilon_label = tk.Label(
            stats_frame,
            text=f"Epsilon: {self.agent.epsilon:.3f}",
            bg="#0d1117",
            fg="#8b949e",
            font=("Arial", 11)
        )
        self.epsilon_label.pack(pady=3)
        
        # Avg reward
        self.reward_label = tk.Label(
            stats_frame,
            text="Avg Reward: N/A",
            bg="#0d1117",
            fg="#8b949e",
            font=("Arial", 11)
        )
        self.reward_label.pack(pady=3)
        
        # Key status
        self.key_label = tk.Label(
            stats_frame,
            text="🔑 Key: Not Collected",
            bg="#0d1117",
            fg="#f0883e",
            font=("Arial", 11, "bold")
        )
        self.key_label.pack(pady=3)
        
        # Progress bar
        self.progress = ttk.Progressbar(
            stats_frame,
            length=220,
            mode='determinate'
        )
        self.progress.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.frame_controls,
            text="🟢 Ready",
            bg="#161b22",
            fg="#7ee787",
            font=("Arial", 12, "bold"),
            wraplength=220
        )
        self.status_label.pack(pady=15)
        
        # Training parameters frame
        param_frame = tk.Frame(self.frame_controls, bg="#0d1117", relief="solid", borderwidth=1)
        param_frame.pack(pady=10, padx=15, fill="x")
        
        tk.Label(
            param_frame,
            text="⚙️ Training Settings",
            bg="#0d1117",
            fg="#79c0ff",
            font=("Arial", 11, "bold")
        ).pack(pady=5)
        
        # Episodes input
        episodes_frame = tk.Frame(param_frame, bg="#0d1117")
        episodes_frame.pack(pady=5)
        
        tk.Label(
            episodes_frame,
            text="Episodes:",
            bg="#0d1117",
            fg="#8b949e",
            font=("Arial", 10)
        ).pack(side="left", padx=5)
        
        self.episodes_entry = tk.Entry(
            episodes_frame,
            width=10,
            font=("Arial", 10),
            bg="#21262d",
            fg="#c9d1d9",
            insertbackground="#c9d1d9"
        )
        self.episodes_entry.insert(0, "2000")
        self.episodes_entry.pack(side="left", padx=5)
        
        # Buttons
        button_frame = tk.Frame(self.frame_controls, bg="#161b22")
        button_frame.pack(pady=20)
        
        self.train_btn = tk.Button(
            button_frame,
            text="🚀 Train Agent",
            width=20,
            height=2,
            bg="#238636",
            fg="white",
            font=("Arial", 11, "bold"),
            activebackground="#2ea043",
            cursor="hand2",
            command=self.train_agent_threaded
        )
        self.train_btn.pack(pady=8)
        
        self.run_btn = tk.Button(
            button_frame,
            text="▶️ Run Agent",
            width=20,
            height=2,
            bg="#1f6feb",
            fg="white",
            font=("Arial", 11, "bold"),
            activebackground="#388bfd",
            cursor="hand2",
            command=self.run_agent
        )
        self.run_btn.pack(pady=8)
        
        self.reset_btn = tk.Button(
            button_frame,
            text="🔄 Reset Agent",
            width=20,
            height=2,
            bg="#da3633",
            fg="white",
            font=("Arial", 11, "bold"),
            activebackground="#f85149",
            cursor="hand2",
            command=self.reset_agent
        )
        self.reset_btn.pack(pady=8)
        
        # Legend
        legend_frame = tk.Frame(self.frame_controls, bg="#0d1117", relief="solid", borderwidth=1)
        legend_frame.pack(pady=10, padx=15, fill="x")
        
        tk.Label(
            legend_frame,
            text="📖 Legend",
            bg="#0d1117",
            fg="#79c0ff",
            font=("Arial", 11, "bold")
        ).pack(pady=5)
        
        legends = [
            ("🤖", "Agent", "#1f6feb"),
            ("🔑", "Key (+15)", "#f0883e"),
            ("🎯", "Goal (+50)", "#4caf50"),
            ("⚠️", "Pit (-20)", "#f44336"),
            ("🚧", "Obstacle", "#8b4513"),
            ("⬜", "Empty (-1)", "#21262d")
        ]
        
        for emoji, text, color in legends:
            lf = tk.Frame(legend_frame, bg="#0d1117")
            lf.pack(pady=2)
            tk.Label(
                lf,
                text=f"{emoji} {text}",
                bg="#0d1117",
                fg="#8b949e",
                font=("Arial", 9)
            ).pack(side="left", padx=10)
        
        # Info label
        info_label = tk.Label(
            self.frame_controls,
            text="💡 Agent must collect\nthe key before reaching\nthe goal!",
            bg="#161b22",
            fg="#f0883e",
            font=("Arial", 9, "italic"),
            justify="center"
        )
        info_label.pack(pady=10)
        
        # Draw initial grid
        self.draw_grid()
        self.window.mainloop()
    
    # --------------------------------------------------------
    def draw_grid(self):
        self.canvas.delete("all")
        size = 100
        
        for i in range(6):
            for j in range(6):
                x1, y1 = j * size, i * size
                x2, y2 = x1 + size, y1 + size
                
                # Determine color
                if (i, j) == self.env.goal:
                    color = "#2ea043"  # Green
                    emoji = "🎯"
                elif (i, j) == self.env.key_pos:
                    if self.env.has_key:
                        color = "#21262d"  # Dark if collected
                        emoji = ""
                    else:
                        color = "#f0883e"  # Orange for key
                        emoji = "🔑"
                elif (i, j) in [self.env.pit1, self.env.pit2]:
                    color = "#da3633"  # Red
                    emoji = "⚠️"
                elif (i, j) in self.env.obstacles:
                    color = "#6e4c1e"  # Brown
                    emoji = "🚧"
                else:
                    color = "#21262d"  # Dark gray
                    emoji = ""
                
                # Draw cell
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline="#30363d",
                    width=2
                )
                
                # Draw emoji
                if emoji:
                    self.canvas.create_text(
                        x1 + size/2, y1 + size/2,
                        text=emoji,
                        font=("Arial", 28),
                        fill="white"
                    )
                
                # Draw Q-values if trained (show arrows for best actions)
                if self.total_episodes > 50:
                    has_key = int(self.env.has_key)
                    q_vals = self.agent.q[i, j, has_key]
                    max_q = np.max(q_vals)
                    if max_q > 0.1:
                        action = np.argmax(q_vals)
                        cx, cy = x1 + size/2, y1 + size/2
                        
                        arrows = {
                            0: (cx, cy - 15, cx, cy - 35),      # up
                            1: (cx, cy + 15, cx, cy + 35),      # down
                            2: (cx - 15, cy, cx - 35, cy),      # left
                            3: (cx + 15, cy, cx + 35, cy)       # right
                        }
                        
                        excluded = [self.env.goal, self.env.key_pos, self.env.pit1, self.env.pit2]
                        if action in arrows and (i, j) not in excluded:
                            self.canvas.create_line(
                                *arrows[action],
                                arrow=tk.LAST,
                                fill="#58a6ff",
                                width=2
                            )
        
        # Draw agent
        ax, ay = self.env.state
        agent_color = "#ffd700" if self.env.has_key else "#1f6feb"
        self.canvas.create_oval(
            ay * size + 25, ax * size + 25,
            ay * size + 75, ax * size + 75,
            fill=agent_color,
            outline="#58a6ff",
            width=3
        )
        
        # Draw agent emoji
        self.canvas.create_text(
            ay * size + 50, ax * size + 50,
            text="🤖",
            font=("Arial", 26)
        )
        
        # Update key status label
        if self.env.has_key:
            self.key_label.config(text="🔑 Key: ✅ Collected", fg="#7ee787")
        else:
            self.key_label.config(text="🔑 Key: ❌ Not Collected", fg="#f0883e")
    
    # --------------------------------------------------------
    def train_agent_threaded(self):
        if self.is_training:
            return
        thread = threading.Thread(target=self.train_agent)
        thread.daemon = True
        thread.start()
    
    def train_agent(self):
        self.is_training = True
        self.train_btn.config(state="disabled")
        
        try:
            episodes = int(self.episodes_entry.get())
        except:
            episodes = 2000
            
        self.status_label.config(text="🟡 Training...", fg="#f0883e")
        episode_rewards = []
        
        for ep in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 100
            
            while not done and steps < max_steps:
                action = self.agent.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.agent.learn(state, action, reward, next_state)
                total_reward += reward
                state = next_state
                steps += 1
            
            episode_rewards.append(total_reward)
            self.agent.decay_epsilon()
            self.total_episodes += 1
            
            if ep % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                self.episode_label.config(text=f"Episodes: {self.total_episodes}")
                self.epsilon_label.config(text=f"Epsilon: {self.agent.epsilon:.3f}")
                self.reward_label.config(text=f"Avg Reward: {avg_reward:.2f}")
                self.progress['value'] = (ep / episodes) * 100
                self.status_label.config(
                    text=f"🟡 Training: {ep}/{episodes}"
                )
                if ep % 50 == 0:
                    self.draw_grid()
                self.window.update()
        
        self.progress['value'] = 100
        self.status_label.config(text="🟢 Training Complete!", fg="#7ee787")
        self.draw_grid()
        self.train_btn.config(state="normal")
        self.is_training = False
    
    # --------------------------------------------------------
    def run_agent(self):
        if self.is_training:
            return
            
        state = self.env.reset()
        done = False
        steps = 0
        max_steps = 100
        
        self.status_label.config(text="▶️ Running...", fg="#58a6ff")
        
        while not done and steps < max_steps:
            self.draw_grid()
            self.window.update()
            time.sleep(0.3)
            
            x, y, has_key = state
            action = np.argmax(self.agent.q[x, y, has_key])
            next_state, reward, done = self.env.step(action)
            state = next_state
            steps += 1
        
        self.draw_grid()
        
        if self.env.state == self.env.goal and self.env.has_key:
            self.status_label.config(text="🎉 Goal Reached!", fg="#7ee787")
        elif self.env.state in [self.env.pit1, self.env.pit2]:
            self.status_label.config(text="💀 Fell in Pit!", fg="#f85149")
        else:
            self.status_label.config(text="⏱️ Max Steps Reached", fg="#f0883e")
    
    # --------------------------------------------------------
    def reset_agent(self):
        if self.is_training:
            return
            
        self.agent = Agent(self.env)
        self.total_episodes = 0
        self.total_rewards = []
        self.episode_label.config(text="Episodes: 0")
        self.epsilon_label.config(text=f"Epsilon: {self.agent.epsilon:.3f}")
        self.reward_label.config(text="Avg Reward: N/A")
        self.progress['value'] = 0
        self.status_label.config(text="🟢 Reset Complete", fg="#7ee787")
        self.env.reset()
        self.draw_grid()

# --------------------------------------------------------
# RUN
# --------------------------------------------------------
if __name__ == "__main__":
    GameGUI()