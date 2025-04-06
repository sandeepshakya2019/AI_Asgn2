# import subprocess
# import sys

# subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio", "matplotlib"])

import numpy as np
import time
import imageio
import matplotlib.pyplot as plt


class HillClimbingTSP:

    def __init__(self, distMat, num_restarts=10, iterationMax=1000):
        self.distMatrix = distMat
        self.citiesNum = len(distMat)
        self.restartNum = num_restarts
        self.iterationMax = iterationMax

    def totlaDist(self, tour):
        return sum(self.distMatrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1)) + \
               self.distMatrix[tour[-1], tour[0]]

    def swapCities(self, tour):
        newTour = tour.copy()
        i, j = np.random.choice(len(tour), 2, replace=False)
        newTour[i], newTour[j] = newTour[j], newTour[i]
        return newTour

    def hillAlgo(self):
        bestTour, bestDistance = None, float('inf')
        allFrame = []
        startTime = time.time()

        for _ in range(self.restartNum):
            currTour = np.random.permutation(self.citiesNum)
            currDisst = self.totlaDist(currTour)

            for _ in range(self.iterationMax):
                if time.time() - startTime > 120:
                    break

                newTourr = self.swapCities(currTour)
                newDist = self.totlaDist(newTourr)
                
                if newDist < currDisst:
                    currTour, currDisst = newTourr, newDist
                
                allFrame.append(self.plot_tour(currTour))

            if currDisst < bestDistance:
                bestTour, bestDistance = currTour, currDisst

        self.gifSave(allFrame, "hill_tsp.gif")
        return bestTour, bestDistance, time.time() - startTime

    def plot_tour(self, tour):
        fig, ax = plt.subplots()
        cities = np.random.rand(self.citiesNum, 2)
        citiesOrder = cities[tour]

        ax.plot(citiesOrder[:, 0], citiesOrder[:, 1], 'o-', markersize=8)
        ax.plot([citiesOrder[-1, 0], citiesOrder[0, 0]], [citiesOrder[-1, 1], citiesOrder[0, 1]], 'r--')
        plt.savefig("hill_tour.png")
        plt.close(fig)
        return imageio.imread("hill_tour.png")

    def gifSave(self, frames, filename):
        if frames:
            imageio.mimsave(filename, frames, duration=0.2)

if __name__ == "__main__":
    num_cities = 20
    distance_matrix = np.random.randint(10, 100, size=(num_cities, num_cities))
    np.fill_diagonal(distance_matrix, 0)

    solver = HillClimbingTSP(distance_matrix)
    bestTour, bestDistance, totalTime = solver.hillAlgo()

    print("Best Tour : ", bestTour)
    print("Best Distance : ", bestDistance)
    print("Time Taken : ", totalTime, "seconds")
