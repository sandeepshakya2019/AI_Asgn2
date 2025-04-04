import subprocess
import time
import csv
import re
import matplotlib.pyplot as plt

# File paths
bnb_script = "bnb.py"
ida_script = "ida.py"
csv_filename = "exec_time.csv"
plot_filename = "exec_time_plot.png"

num_runs = 5
results = []
iterations = []
bnb_times = []
ida_times = []

def run_script(script_name):
    """Runs a script and extracts execution time using regex."""
    start_time = time.time()
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    elapsed_time = time.time() - start_time

    output = result.stdout.strip()

    # Extract execution time using regex
    match = re.search(r"Exec Time\s*:\s*([\d\.]+)\s*seconds", output)
    
    if match:
        exec_time = float(match.group(1))
    else:
        exec_time = elapsed_time  # Use total runtime as fallback
        print(f"⚠️ Warning: Could not extract execution time from {script_name}, using actual elapsed time.")

    return exec_time

# Run experiments
for i in range(1, num_runs + 1):
    print(f"Running test {i}/{num_runs}...")

    bnb_time = run_script(bnb_script)
    ida_time = run_script(ida_script)

    iterations.append(i)
    bnb_times.append(bnb_time)
    ida_times.append(ida_time)

    results.append([i, f"{bnb_time:.6e}", f"{ida_time:.6e}"])

# Save results to CSV
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Iteration", "BnB Exec Time (s)", "IDA* Exec Time (s)"])
    writer.writerows(results)

print(f"Execution times saved in {csv_filename}")

# Plot execution times
plt.figure(figsize=(8, 5))
plt.plot(iterations, bnb_times, marker='o', linestyle='-', color='b', label="BnB Execution Time")
plt.plot(iterations, ida_times, marker='s', linestyle='--', color='r', label="IDA* Execution Time")

plt.xlabel("Iteration")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time Comparison: BnB vs IDA*")
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig(plot_filename)
# plt.show()

print(f"Plot saved as {plot_filename}")
