import subprocess
import time
import csv
import matplotlib.pyplot as plt

scripts = ["hillClimbing.py", "simulatedAnnealing.py"]
csvFile = "execTime.csv"
numRuns = 5
results = []

def runScript(scriptName):
    startTime = time.time()
    try:
        result = subprocess.run(["python", scriptName], capture_output=True, text=True, timeout=300)
        elapsedTime = time.time() - startTime

        output = result.stdout.strip()
        
        try:
            timeTaken = float(output.split("Time Taken : ")[1].split(" seconds")[0])
        except (IndexError, ValueError):
            timeTaken = elapsedTime 

        return timeTaken
    except subprocess.TimeoutExpired:
        return float("inf")
    except Exception:
        return float("inf")

for i in range(numRuns):
    row = [i + 1]
    for script in scripts:
        timeTaken = runScript(script)
        row.append(timeTaken)
    results.append(row)

with open(csvFile, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration"] + [f"{script} Execution Time (s)" for script in scripts])
    writer.writerows(results)

iterations = [row[0] for row in results]
hillTimes = [row[1] for row in results]
saTimes = [row[2] for row in results]

plt.figure(figsize=(8, 5))
plt.plot(iterations, hillTimes, marker='o', linestyle='-', label='Hill Climbing')
plt.plot(iterations, saTimes, marker='s', linestyle='-', label='Simulated Annealing')
plt.xlabel("Iteration")
plt.ylabel("Execution Time (s)")
plt.title("Comparison of Execution Times")
plt.legend()
plt.grid()
plt.savefig("executionTimePlot.png")
plt.show()

print(f"Execution times saved in {csvFile}")
print("Comparison plot saved as executionTimePlot.png")
