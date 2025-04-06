import time
import numpy as np
import matplotlib.pyplot as plt
import imageio

class SATSP:
    
    def __init__(self, distMat, temp=1000, coolRate=0.995, minTemp=1):
        self.distMat = distMat
        self.n = len(distMat)
        self.temp = temp
        self.coolRate = coolRate
        self.minTemp = minTemp

    def totalDist(self, tour):
        return sum(self.distMat[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + self.distMat[tour[-1], tour[0]]

    def swap(self, tour):
        newTour = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        newTour[i], newTour[j] = newTour[j], newTour[i]
        return newTour

    def solve(self):
        tour = np.random.permutation(self.n)
        dist = self.totalDist(tour)
        bestTour, bestDist = tour, dist
        frames = []
        
        startTime = time.time()
        maxTime = 60  

        while self.temp > self.minTemp:
            if time.time() - startTime > maxTime:
                break

            newTour = self.swap(tour)
            newDist = self.totalDist(newTour)
            
            delta = newDist - dist
            if delta < 0 or np.random.rand() < np.exp(-delta / self.temp):
                tour, dist = newTour, newDist
            
            if dist < bestDist:
                bestTour, bestDist = tour, dist

            frames.append(self.plotTour(tour))
            self.temp *= self.coolRate

        self.saveGif(frames, "sa_tsp.gif")
        return bestTour, bestDist, time.time() - startTime

    def plotTour(self, tour):
        fig, ax = plt.subplots()
        cities = np.random.rand(self.n, 2)
        ordered = cities[tour]

        ax.plot(ordered[:, 0], ordered[:, 1], 'o-', markersize=8, label="Path")
        ax.plot([ordered[-1, 0], ordered[0, 0]], [ordered[-1, 1], ordered[0, 1]], 'r--')
        ax.set_title("TSP - Simulated Annealing")
        ax.legend()
        
        filename = "sa_tour.png"
        plt.savefig(filename)
        plt.close(fig)
        return imageio.imread(filename)

    def saveGif(self, frames, filename):
        if frames:
            imageio.mimsave(filename, frames, duration=0.2)

if __name__ == "__main__":
    n = 20
    distMat = np.random.randint(10, 100, size=(n, n))
    np.fill_diagonal(distMat, 0)

    solver = SATSP(distMat)
    bestTour, bestDist, timeTaken = solver.solve()

    print("Best Tour:", bestTour)
    print("Best Distance:", bestDist)
    print("Time Taken:", timeTaken, "seconds")