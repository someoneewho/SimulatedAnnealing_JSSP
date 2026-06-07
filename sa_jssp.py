"""
Simulated Annealing for the Job Shop Scheduling Problem (JSSP)
==============================================================
CSE480/586 Term Project - Team 8

"""

import random
import time
import math
import statistics
import os
import sys
import csv
import json

# ============================================================
# Best Known Solutions (BKS) from literature
# Sources: thomasWeise/jsspInstancesAndResults, BRKGA (Goncalves & Resende, 2013)
# ============================================================
BKS = {
    "dmu01": 2563,  "dmu02": 2706,  "dmu03": 2731,  "dmu04": 2669,  "dmu05": 2749,
    "dmu06": 3244,  "dmu07": 3046,  "dmu08": 3188,  "dmu09": 3092,  "dmu10": 2984,
    "dmu11": 3430,  "dmu12": 3492,  "dmu13": 3681,  "dmu14": 3394,  "dmu15": 3343,
    "dmu16": 3751,  "dmu17": 3814,  "dmu18": 3844,  "dmu19": 3765,  "dmu20": 3710,
    "dmu51": 4196,  "dmu52": 4311,  "dmu53": 4390,  "dmu54": 4362,  "dmu55": 4270,
    "dmu56": 4941,  "dmu57": 4663,  "dmu58": 4708,  "dmu59": 4619,  "dmu60": 4739,
}

# ============================================================
# Parameter Configurations
# T0 is set to "auto" — will be calibrated per-instance so that
# initial acceptance probability ≈ 0.8 (as per lecture slide 9).
# ============================================================
CONFIGS = {
    "baseline": {
        "T0": "auto",       # calibrated per-instance
        "alpha": 0.95,
        "T_min": 0.01,
        "L": 200,
    },
    "improved": {
        "T0": "auto",       # calibrated per-instance
        "alpha": 0.97,
        "T_min": 0.01,
        "L": 500,
    },
}

# 5 independent runs per instance (Phase 3 requirement)
SEEDS = [42, 123, 456, 789, 1011]

# Convergence log interval: record best fitness every N function evaluations
CONVERGENCE_LOG_INTERVAL = 1000

# ============================================================
# Filename-to-DMU mapping (from report Table 1)
# ============================================================
FILENAME_TO_DMU = {
    "rcmax_20_15_4":  "dmu01", "rcmax_20_15_10": "dmu02",
    "rcmax_20_15_5":  "dmu03", "rcmax_20_15_8":  "dmu04",
    "rcmax_20_15_1":  "dmu05", "rcmax_20_20_6":  "dmu06",
    "rcmax_20_20_4":  "dmu07", "rcmax_20_20_7":  "dmu08",
    "rcmax_20_20_8":  "dmu09", "rcmax_20_20_5":  "dmu10",
    "rcmax_30_15_9":  "dmu11", "rcmax_30_15_10": "dmu12",
    "rcmax_30_15_5":  "dmu13", "rcmax_30_15_4":  "dmu14",
    "rcmax_30_15_1":  "dmu15", "rcmax_30_20_7":  "dmu16",
    "rcmax_30_20_10": "dmu17", "rcmax_30_20_9":  "dmu18",
    "rcmax_30_20_8":  "dmu19", "rcmax_30_20_2":  "dmu20",
    "cscmax_30_15_2": "dmu51", "cscmax_30_15_9": "dmu52",
    "cscmax_30_15_10":"dmu53", "cscmax_30_15_5": "dmu54",
    "cscmax_30_15_6": "dmu55", "cscmax_30_20_9": "dmu56",
    "cscmax_30_20_7": "dmu57", "cscmax_30_20_3": "dmu58",
    "cscmax_30_20_6": "dmu59", "cscmax_30_20_4": "dmu60",
}


# ============================================================
# Instance Parser
# ============================================================
def parse_instance(filepath):
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    first_line = lines[0].split()
    n_jobs = int(first_line[0])
    n_machines = int(first_line[1])
    jobs = []
    for i in range(1, n_jobs + 1):
        values = list(map(int, lines[i].split()))
        operations = []
        for k in range(0, len(values), 2):
            operations.append((values[k], values[k + 1]))
        jobs.append(operations)
    return n_jobs, n_machines, jobs


# ============================================================
# Solution Representation & Fitness
# ============================================================
def generate_random_solution(n_jobs, n_machines):
    solution = []
    for j in range(n_jobs):
        solution.extend([j] * n_machines)
    random.shuffle(solution)
    return solution


def compute_makespan(solution, jobs, n_jobs, n_machines):
    job_op_count = [0] * n_jobs
    job_ready = [0] * n_jobs
    machine_ready = [0] * n_machines
    for job_id in solution:
        op_idx = job_op_count[job_id]
        machine, proc_time = jobs[job_id][op_idx]
        start = max(job_ready[job_id], machine_ready[machine])
        finish = start + proc_time
        job_ready[job_id] = finish
        machine_ready[machine] = finish
        job_op_count[job_id] = op_idx + 1
    return max(job_ready)


def get_neighbor(solution):
    n = len(solution)
    neighbor = solution[:]
    while True:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j and neighbor[i] != neighbor[j]:
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            return neighbor


# ============================================================
# T0 Calibration (Lecture Slide 9)
# "Initial temperature T0: calibrated so initial acceptance ≈ 0.8"
#
# Method: Sample random neighbors, measure average worsening delta,
# then solve  exp(-avg_delta / T0) = 0.8  =>  T0 = -avg_delta / ln(0.8)
# ============================================================
def calibrate_T0(n_jobs, n_machines, jobs, target_acceptance=0.8, n_samples=500, seed=0):
    """
    Calibrate initial temperature so that the acceptance probability
    of a random worsening move is approximately target_acceptance.
    """
    random.seed(seed)
    solution = generate_random_solution(n_jobs, n_machines)
    current_cost = compute_makespan(solution, jobs, n_jobs, n_machines)

    worsening_deltas = []
    for _ in range(n_samples):
        neighbor = get_neighbor(solution)
        neighbor_cost = compute_makespan(neighbor, jobs, n_jobs, n_machines)
        delta = neighbor_cost - current_cost
        if delta > 0:
            worsening_deltas.append(delta)

    if not worsening_deltas:
        return 1000.0  # fallback

    avg_delta = statistics.mean(worsening_deltas)
    T0 = -avg_delta / math.log(target_acceptance)
    return round(T0, 2)


# ============================================================
# Simulated Annealing (with convergence tracking)
# ============================================================
def simulated_annealing(n_jobs, n_machines, jobs, config, seed, T0_calibrated):
    """
    Run one SA instance. Returns: best_makespan, best_solution, stats_dict, convergence_log
    convergence_log = list of (FE_count, best_makespan) for plotting.
    """
    random.seed(seed)

    T0 = T0_calibrated
    alpha = config["alpha"]
    T_min = config["T_min"]
    L = config["L"]

    # Generate initial random solution
    current = generate_random_solution(n_jobs, n_machines)
    current_cost = compute_makespan(current, jobs, n_jobs, n_machines)

    best = current[:]
    best_cost = current_cost

    T = T0
    total_FEs = 0          # function evaluations (= makespan computations)
    best_found_at_FE = 0
    improvements = 0
    worse_accepted = 0

    # Convergence log: record best fitness at regular intervals
    convergence_log = [(0, best_cost)]
    next_log_point = CONVERGENCE_LOG_INTERVAL

    # Main SA loop
    while T > T_min:
        for _ in range(L):
            total_FEs += 1

            neighbor = get_neighbor(current)
            neighbor_cost = compute_makespan(neighbor, jobs, n_jobs, n_machines)
            delta = neighbor_cost - current_cost

            if delta <= 0:
                current = neighbor
                current_cost = neighbor_cost
                improvements += 1
                if neighbor_cost < best_cost:
                    best = neighbor[:]
                    best_cost = neighbor_cost
                    best_found_at_FE = total_FEs
            else:
                r = random.random()
                if r < math.exp(-delta / T):
                    current = neighbor
                    current_cost = neighbor_cost
                    worse_accepted += 1

            # Log convergence at regular intervals
            if total_FEs >= next_log_point:
                convergence_log.append((total_FEs, best_cost))
                next_log_point += CONVERGENCE_LOG_INTERVAL

        T *= alpha

    # Final log point
    convergence_log.append((total_FEs, best_cost))

    stats = {
        "total_FEs": total_FEs,
        "best_found_at_FE": best_found_at_FE,
        "best_found_at_pct": round(100.0 * best_found_at_FE / total_FEs, 1) if total_FEs > 0 else 0,
        "improvements": improvements,
        "worse_accepted": worse_accepted,
        "T0_used": T0,
    }

    return best_cost, best, stats, convergence_log


# ============================================================
# Schedule Verification
# ============================================================
def decode_schedule(solution, jobs, n_jobs, n_machines):
    job_op_count = [0] * n_jobs
    job_ready = [0] * n_jobs
    machine_ready = [0] * n_machines
    schedule = []
    for job_id in solution:
        op_idx = job_op_count[job_id]
        machine, proc_time = jobs[job_id][op_idx]
        start = max(job_ready[job_id], machine_ready[machine])
        finish = start + proc_time
        schedule.append((job_id, op_idx, machine, start, finish))
        job_ready[job_id] = finish
        machine_ready[machine] = finish
        job_op_count[job_id] = op_idx + 1
    return schedule


def verify_schedule(schedule, jobs, n_jobs, n_machines):
    job_finish = {}
    for job_id, op_idx, machine, start, finish in schedule:
        if op_idx > 0:
            prev_finish = job_finish.get((job_id, op_idx - 1))
            if prev_finish is None:
                return False, f"Job {job_id} op {op_idx}: predecessor not scheduled"
            if start < prev_finish:
                return False, f"Job {job_id} op {op_idx}: precedence violated"
        job_finish[(job_id, op_idx)] = finish
    machine_ops = {}
    for job_id, op_idx, machine, start, finish in schedule:
        if machine not in machine_ops:
            machine_ops[machine] = []
        machine_ops[machine].append((start, finish, job_id, op_idx))
    for m, ops in machine_ops.items():
        ops.sort()
        for i in range(len(ops) - 1):
            if ops[i][1] > ops[i + 1][0]:
                return False, f"Machine {m}: overlap"
    op_count = {}
    for job_id, op_idx, _, _, _ in schedule:
        op_count[job_id] = op_count.get(job_id, 0) + 1
    for j in range(n_jobs):
        if op_count.get(j, 0) != n_machines:
            return False, f"Job {j}: missing operations"
    return True, "Valid"


# ============================================================
# Instance ID Extraction
# ============================================================
def extract_instance_id(filepath):
    basename = os.path.basename(filepath).lower()
    for i in range(1, 81):
        tag = f"dmu{i:02d}"
        if tag in basename:
            return tag
    name_no_ext = os.path.splitext(basename)[0]
    for i in range(1, 81):
        prefix = f"dmu{i:02d}_"
        if name_no_ext.startswith(prefix):
            name_no_ext = name_no_ext[len(prefix):]
            break
    if name_no_ext in FILENAME_TO_DMU:
        return FILENAME_TO_DMU[name_no_ext]
    return None


# ============================================================
# Main Runner
# ============================================================
def run_instance(filepath, config_name="baseline"):
    """Run SA on a single instance with 10 seeds, report full metrics."""
    config = CONFIGS[config_name]
    n_jobs, n_machines, jobs = parse_instance(filepath)
    instance_id = extract_instance_id(filepath)
    basename = os.path.basename(filepath)

    # --- T0 Calibration (slide 9) ---
    if config.get("T0") == "auto":
        T0_calibrated = calibrate_T0(n_jobs, n_machines, jobs)
    else:
        T0_calibrated = config["T0"]

    print(f"\n{'='*70}")
    print(f"Instance: {basename}")
    print(f"Size: {n_jobs} jobs x {n_machines} machines = {n_jobs * n_machines} operations")
    print(f"Config: {config_name} (T0={T0_calibrated} [calibrated], "
          f"alpha={config['alpha']}, T_min={config['T_min']}, L={config['L']})")

    n_steps = int(math.log(config['T_min'] / T0_calibrated) / math.log(config['alpha']))
    total_evals = n_steps * config['L']
    print(f"Temperature steps: ~{n_steps}, Total FEs: ~{total_evals}")
    print(f"Seeds: {SEEDS} ({len(SEEDS)} independent runs)")
    print(f"{'='*70}")

    results = []
    all_convergence = []

    for seed in SEEDS:
        start_time = time.time()
        best_cost, best_sol, stats, conv_log = simulated_annealing(
            n_jobs, n_machines, jobs, config, seed, T0_calibrated
        )
        elapsed = time.time() - start_time

        schedule = decode_schedule(best_sol, jobs, n_jobs, n_machines)
        is_valid, msg = verify_schedule(schedule, jobs, n_jobs, n_machines)

        results.append({
            "seed": seed,
            "makespan": best_cost,
            "time_sec": round(elapsed, 2),
            "valid": is_valid,
            "total_FEs": stats["total_FEs"],
            "best_at_pct": stats["best_found_at_pct"],
        })
        all_convergence.append(conv_log)

        status = "pass" if is_valid else f"FAIL ({msg})"
        print(f"  Seed {seed:>5d}: Makespan = {best_cost:>5d}  |  "
              f"Time = {elapsed:>6.2f}s  |  "
              f"Best at {stats['best_found_at_pct']:>5.1f}% of FEs  |  {status}")

    # ---- Full metrics (slides 11 & 13) ----
    makespans = [r["makespan"] for r in results]
    times = [r["time_sec"] for r in results]
    best_ms = min(makespans)
    worst_ms = max(makespans)
    avg_ms = statistics.mean(makespans)
    median_ms = statistics.median(makespans)
    std_ms = statistics.stdev(makespans) if len(makespans) > 1 else 0
    cv = (std_ms / avg_ms * 100) if avg_ms > 0 else 0  # Coefficient of Variation
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0

    print(f"\n  Solution Quality Metrics (across {len(SEEDS)} runs):")
    print(f"    Best:     {best_ms}")
    print(f"    Mean:     {avg_ms:.1f}")
    print(f"    Median:   {median_ms:.1f}")
    print(f"    Worst:    {worst_ms}")
    print(f"    Std Dev:  {std_ms:.1f}")
    print(f"    CV:       {cv:.2f}%")
    print(f"  Non-Functional Metrics:")
    print(f"    Avg Time: {avg_time:.2f}s +/- {std_time:.2f}s")
    print(f"    T0 used:  {T0_calibrated}")

    # BKS comparison
    rpd = None
    bks_val = None
    if instance_id and instance_id in BKS:
        bks_val = BKS[instance_id]
        rpd = 100.0 * (best_ms - bks_val) / bks_val
        rpd_avg = 100.0 * (avg_ms - bks_val) / bks_val
        print(f"  BKS Comparison:")
        print(f"    BKS:       {bks_val}")
        print(f"    RPD best:  {rpd:.2f}%")
        print(f"    RPD avg:   {rpd_avg:.2f}%")

    return {
        "instance": basename,
        "instance_id": instance_id or "N/A",
        "size": f"{n_jobs}x{n_machines}",
        "config": config_name,
        "T0_calibrated": T0_calibrated,
        "best": best_ms,
        "mean": round(avg_ms, 1),
        "median": round(median_ms, 1),
        "worst": worst_ms,
        "std": round(std_ms, 1),
        "cv_pct": round(cv, 2),
        "bks": bks_val if bks_val else "N/A",
        "rpd_best": round(rpd, 2) if rpd is not None else "N/A",
        "rpd_avg": round(100.0 * (avg_ms - bks_val) / bks_val, 2) if bks_val else "N/A",
        "avg_time": round(avg_time, 2),
        "std_time": round(std_time, 2),
        "total_FEs": results[0]["total_FEs"],
        "convergence": all_convergence,
    }


def save_results_csv(all_results, output_path):
    """Save summary results to CSV."""
    if not all_results:
        return
    fieldnames = [
        "instance", "instance_id", "size", "config", "T0_calibrated",
        "best", "mean", "median", "worst", "std", "cv_pct",
        "bks", "rpd_best", "rpd_avg",
        "avg_time", "std_time", "total_FEs"
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            filtered = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered)
    print(f"\nResults saved to: {output_path}")


def save_convergence_csv(all_results, output_dir):
    """Save convergence data for plotting (fitness vs FEs)."""
    for result in all_results:
        if "convergence" not in result:
            continue
        inst = result["instance"].replace(".txt", "")
        cfg = result["config"]
        path = os.path.join(output_dir, f"convergence_{inst}_{cfg}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["FE"] + [f"seed_{s}" for s in SEEDS]
            writer.writerow(header)
            # Align all runs to the same FE checkpoints
            all_conv = result["convergence"]
            max_len = max(len(c) for c in all_conv)
            for idx in range(max_len):
                row = []
                fe_val = None
                for conv in all_conv:
                    if idx < len(conv):
                        fe_val = conv[idx][0]
                        row.append(conv[idx][1])
                    else:
                        row.append(conv[-1][1])
                if fe_val is not None:
                    writer.writerow([fe_val] + row)
    print(f"Convergence data saved to: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python sa_jssp.py <instance_file>              # Single instance")
        print("  python sa_jssp.py <folder>                     # All .txt files")
        print("  python sa_jssp.py <path> --config improved     # Use improved params")
        print("  python sa_jssp.py <path> --config both         # Run both configs")
        sys.exit(1)

    path = sys.argv[1]
    config_name = "baseline"
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_name = sys.argv[idx + 1]

    configs_to_run = ["baseline", "improved"] if config_name == "both" else [config_name]

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")])
    else:
        print(f"Error: {path} not found.")
        sys.exit(1)

    print(f"Found {len(files)} instance(s)")
    print(f"Configs: {configs_to_run}")
    print(f"Seeds: {SEEDS} ({len(SEEDS)} runs per instance)")
    print(f"Total runs: {len(files) * len(configs_to_run) * len(SEEDS)}")

    all_results = []
    for cfg in configs_to_run:
        for filepath in files:
            result = run_instance(filepath, cfg)
            all_results.append(result)

    # Save outputs
    out_dir = "/home/claude" if "/uploads" in path else (
        os.path.dirname(path) if os.path.isfile(path) else path
    )

    save_results_csv(all_results, os.path.join(out_dir, "sa_results.csv"))
    save_convergence_csv(all_results, out_dir)

    # Final summary table
    print(f"\n{'='*105}")
    print(f"{'FINAL SUMMARY TABLE':^105}")
    print(f"{'='*105}")
    print(f"{'Instance':<25} {'Cfg':<9} {'T0':>7} {'Best':>6} {'Mean':>7} {'Med':>7} "
          f"{'Std':>6} {'CV%':>5} {'BKS':>5} {'RPD%':>6} {'Time':>7} {'FEs':>8}")
    print(f"{'-'*105}")

    for r in all_results:
        bks = str(r['bks']) if r['bks'] != "N/A" else " N/A"
        rpd = f"{r['rpd_best']}%" if r['rpd_best'] != "N/A" else " N/A"
        print(f"{r['instance']:<25} {r['config']:<9} {r['T0_calibrated']:>7.1f} "
              f"{r['best']:>6} {r['mean']:>7} {r['median']:>7} "
              f"{r['std']:>6} {r['cv_pct']:>5} {bks:>5} {rpd:>6} "
              f"{r['avg_time']:>6.1f}s {r['total_FEs']:>8}")


if __name__ == "__main__":
    main()
