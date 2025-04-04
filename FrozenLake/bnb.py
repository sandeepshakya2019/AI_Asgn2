# Please Uncoment theis code if no module found
# import subprocess
# import sys

# subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
# subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import numpy as np
import time
import heapq
import cv2
import imageio
import gymnasium as gym
import matplotlib.pyplot as plt


class BranchAndBoundSolver:

    def __init__(self, env):
        self.env = env
        self.startState = 0
        self.goalState = env.observation_space.n - 1
        self.bestCost = float("inf")
        self.bestPath = None
        self.execTime = None

    def findPath(self):
        startTime = time.perf_counter()
        queue = [(0, [self.startState])]
        heapq.heapify(queue)

        while queue:
            cost, path = heapq.heappop(queue)
            currentState = path[-1]

            if currentState == self.goalState:
                self.bestCost = cost
                self.bestPath = path
                self.execTime = time.perf_counter() - startTime
                return self.bestPath, self.bestCost, self.execTime

            for action in range(self.env.action_space.n):
                for prob, next_state, _, _ in self.env.unwrapped.P[currentState][action]:
                    if prob > 0 and next_state not in path:
                        new_cost = cost + 1
                        if new_cost < self.bestCost:
                            heapq.heappush(queue, (new_cost, path + [next_state]))

        self.execTime = time.perf_counter() - startTime
        return None, float("inf"), self.execTime

    def genrateGif(self, path, filename="bnbFrozen.gif"):
        frames = []
        prevState = None
        trail = []

        for idx, state in enumerate(path):
            trail.append(state)
            for pulse in range(1, 4):  # Agent pulse animation
                frame = self.renderFrLake(state, prevState, trail, step_num=idx, pulse_radius=25 + 5 * pulse)
                prevState = state
                large_frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
                frames.append(large_frame)

        # Blink final state
        for i in range(5):
            frame = self.renderFrLake(path[-1], path[-2], trail, step_num=len(path)-1, blink=(i % 2 == 0))
            large_frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_NEAREST)
            frames.append(large_frame)

        imageio.mimsave(filename, frames, duration=0.5)
        print(f"GIF saved as {filename}")

        cv2.imshow("BnB Frozen Lake Solution", frames[-1])
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    def renderFrLake(self, agent_pos, prev_pos=None, trail=None, step_num=None, pulse_radius=30, blink=False):
        size = int(np.sqrt(self.env.observation_space.n))
        lake_map = self.env.unwrapped.desc
        img = np.ones((size * 100, size * 100, 3), dtype=np.uint8) * 255

        color_map = {
            b"S": (0, 255, 0),
            b"F": (200, 200, 200),
            b"H": (0, 0, 0),
            b"G": (255, 215, 0)
        }

        # Draw tiles and borders
        for i in range(size):
            for j in range(size):
                tile = lake_map[i][j]
                color = color_map[tile]
                x1, y1 = i * 100, j * 100
                x2, y2 = x1 + 100, y1 + 100
                img[x1:x2, y1:y2] = color
                cv2.rectangle(img, (y1, x1), (y2, x2), (50, 50, 50), 1)

        # Trail
        if trail:
            for idx, state in enumerate(trail[:-1]):
                x, y = divmod(state, size)
                fade = 200 - int((idx / len(trail)) * 150)
                color = (fade, 50, 255)
                center = (y * 100 + 50, x * 100 + 50)
                cv2.circle(img, center, 12, color, -1)
                cv2.putText(img, str(idx), (center[0] - 10, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Agent
        agent_x, agent_y = divmod(agent_pos, size)
        agent_center = (agent_y * 100 + 50, agent_x * 100 + 50)
        agent_color = (0, 0, 255) if blink else (255, 0, 0)
        cv2.circle(img, agent_center, pulse_radius, agent_color, -1)

        # Movement arrow
        if prev_pos is not None:
            prev_x, prev_y = divmod(prev_pos, size)
            start = (prev_y * 100 + 50, prev_x * 100 + 50)
            end = (agent_y * 100 + 50, agent_x * 100 + 50)
            img = cv2.arrowedLine(img, start, end, (0, 0, 255), 3, tipLength=0.3)

        return img


env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

bnb_solver = BranchAndBoundSolver(env)
solution_path, solution_cost, execTime = bnb_solver.findPath()

if solution_path:
    print("Optimal Path:", solution_path)
    print("Optimal Cost:", solution_cost)
    print(f"Exec Time: {execTime:.6f} seconds")
    bnb_solver.genrateGif(solution_path)
else:
    print("Doesn't find any solution")
