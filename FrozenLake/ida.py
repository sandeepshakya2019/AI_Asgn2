# Please Uncoment theis code if no module found
# import subprocess
# import sys

# subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import gymnasium as gym
import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt
import time

class IDAStarSolver:
    def __init__(self, env):
        self.time_taken = None
        self.env = env
        self.goal = self.env.observation_space.n - 1
        self.start = 0
        self.cost = float("inf")
        self.path = None

    def funHeuristic(self, node):
        n = int(np.sqrt(self.env.observation_space.n))
        x2, y2 = divmod(self.goal, n)
        x1, y1 = divmod(node, n)
        return abs(x1 - x2) + abs(y1 - y2)

    def dfsSolve(self, current, trail, g, bound):
        f = g + self.funHeuristic(current)
        if f > bound:
            return f
        if current == self.goal:
            self.path = trail
            self.cost = g
            return None
        mini = float("inf")
        for action in range(self.env.action_space.n):
            for prob, nxt, _, _ in self.env.unwrapped.P[current][action]:
                if prob > 0 and nxt not in trail:
                    outcome = self.dfsSolve(nxt, trail + [nxt], g + 1, bound)
                    if outcome is None:
                        return None
                    mini = min(mini, outcome)
        return mini

    def findPath(self):
        start_clock = time.perf_counter()
        bound = self.funHeuristic(self.start)
        while True:
            outcome = self.dfsSolve(self.start, [self.start], 0, bound)
            if outcome is None:
                self.time_taken = time.perf_counter() - start_clock
                return self.path, self.cost, self.time_taken
            if outcome == float("inf"):
                self.time_taken = time.perf_counter() - start_clock
                return None, float("inf"), self.time_taken
            bound = outcome

    def frameDraw(self, current, previous=None, step_idx=None):
        dim = int(np.sqrt(self.env.observation_space.n))
        desc = self.env.unwrapped.desc
        canvas = np.ones((dim * 100, dim * 100, 3), dtype=np.uint8) * 255

        colors = {
            b"S": (0, 255, 0),
            b"F": (210, 210, 210),
            b"H": (0, 0, 0),
            b"G": (255, 215, 0)
        }

        for i in range(dim):
            for j in range(dim):
                cell = desc[i][j]
                canvas[i*100:(i+1)*100, j*100:(j+1)*100] = colors[cell]

        x, y = divmod(current, dim)

        if step_idx is not None:
            cv2.putText(canvas, str(step_idx), (y*100 + 30, x*100 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

        if previous is not None:
            px, py = divmod(previous, dim)
            start = (py*100 + 50, px*100 + 50)
            end = (y*100 + 50, x*100 + 50)
            canvas = cv2.arrowedLine(canvas, start, end, (0, 0, 255), 5)

        return canvas

    def gifGenrator(self, steps, filename="idaFrozen.gif"):
        frames = []
        prev = None
        for idx, state in enumerate(steps):
            img = self.frameDraw(state, prev, step_idx=idx)
            img_resized = cv2.resize(img, (400, 400), interpolation=cv2.INTER_NEAREST)
            frames.append(img_resized)
            prev = state
        imageio.mimsave(filename, frames, duration=0.6)
        print(f"Animation saved as '{filename}'")

        cv2.imshow("IDA* Result", frames[-1])
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

solver = IDAStarSolver(environment)
found_path, total_cost, elapsed_time = solver.findPath()

if found_path:
    print("Optimal Path :", found_path)
    print("Step Count :", total_cost)
    print(f"Exec Time : {elapsed_time:.6f} seconds")
    solver.gifGenrator(found_path)
else:
    print("Goal Unreachable")
