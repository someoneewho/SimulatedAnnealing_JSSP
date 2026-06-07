"""
Friedman Test + Holm Post-Hoc for JSSP Algorithm Comparison
============================================================
Compares SA, TS, and GA using Best Objective values across 30 instances.
Following the methodology from Lecture Slide 14 (Terci, 2026).

Usage: python friedman_test.py
"""

from scipy.stats import friedmanchisquare, rankdata
import numpy as np

# ============================================================
# Best Objective Values from Comparison Tables
# Order: [TS, GA, SA] for each instance
# ============================================================
data = {
    # Set A: 20-Job Instances (rcmax)
    "Dmu01": [2870, 3149, 2745],
    "Dmu02": [3040, 2939, 2842],
    "Dmu03": [3023, 3082, 2899],
    "Dmu04": [2988, 3036, 2849],
    "Dmu05": [3178, 3091, 2902],
    "Dmu06": [3843, 3671, 3463],
    "Dmu07": [3518, 3635, 3265],
    "Dmu08": [3654, 3494, 3385],
    "Dmu09": [3549, 3383, 3328],
    "Dmu10": [3455, 3577, 3147],
    # Set B: 30-Job Instances (rcmax)
    "Dmu11": [4108, 3902, 3750],
    "Dmu12": [4215, 3821, 3847],  # GA wins this one
    "Dmu13": [4387, 4372, 4010],
    "Dmu14": [3819, 4089, 3636],
    "Dmu15": [3712, 4171, 3517],
    "Dmu16": [4359, 4508, 4113],
    "Dmu17": [4636, 4721, 4236],
    "Dmu18": [4471, 4590, 4206],
    "Dmu19": [4493, 4661, 4188],
    "Dmu20": [4344, 4479, 4050],
    # Set C: 30-Job Instances (cscmax)
    "Dmu51": [5309, 5519, 5083],
    "Dmu52": [5432, 5744, 5124],
    "Dmu53": [5697, 5556, 5278],
    "Dmu54": [5695, 5572, 5180],
    "Dmu55": [5562, 6022, 5045],
    "Dmu56": [6379, 6250, 5900],
    "Dmu57": [5950, 6311, 5638],
    "Dmu58": [6055, 6189, 5676],
    "Dmu59": [5906, 6159, 5514],
    "Dmu60": [6227, 6450, 5730],
}

algorithms = ["TS", "GA", "SA"]
instances = list(data.keys())
n_instances = len(instances)
n_algorithms = len(algorithms)

# Build arrays
ts_vals = [data[inst][0] for inst in instances]
ga_vals = [data[inst][1] for inst in instances]
sa_vals = [data[inst][2] for inst in instances]

print("=" * 65)
print("FRIEDMAN TEST + HOLM POST-HOC CORRECTION")
print(f"Algorithms: {algorithms}")
print(f"Instances: {n_instances}")
print("=" * 65)

# ============================================================
# Step 1: Compute ranks per instance (1 = best = lowest makespan)
# ============================================================
all_ranks = []
print(f"\n{'Instance':<10} {'TS':>6} {'GA':>6} {'SA':>6}  |  {'R_TS':>5} {'R_GA':>5} {'R_SA':>5}")
print("-" * 55)

for inst in instances:
    values = data[inst]
    ranks = rankdata(values)  # lower value = lower rank = better
    all_ranks.append(ranks)
    print(f"{inst:<10} {values[0]:>6} {values[1]:>6} {values[2]:>6}  |  "
          f"{ranks[0]:>5.1f} {ranks[1]:>5.1f} {ranks[2]:>5.1f}")

all_ranks = np.array(all_ranks)
avg_ranks = all_ranks.mean(axis=0)

print("-" * 55)
print(f"{'Avg Rank':<10} {'':>6} {'':>6} {'':>6}  |  "
      f"{avg_ranks[0]:>5.2f} {avg_ranks[1]:>5.2f} {avg_ranks[2]:>5.2f}")

# ============================================================
# Step 2: Friedman test
# ============================================================
stat, p_value = friedmanchisquare(ts_vals, ga_vals, sa_vals)

print(f"\n{'='*65}")
print(f"FRIEDMAN TEST RESULTS")
print(f"{'='*65}")
print(f"Chi-squared statistic: {stat:.4f}")
print(f"p-value:               {p_value:.6f}")
print(f"Significance level:    alpha = 0.05")

if p_value < 0.05:
    print(f"Result: REJECT H0 — significant difference exists (p < 0.05)")
else:
    print(f"Result: FAIL TO REJECT H0 — no significant difference (p >= 0.05)")

# ============================================================
# Step 3: Holm post-hoc (if Friedman is significant)
# ============================================================
if p_value < 0.05:
    print(f"\n{'='*65}")
    print(f"HOLM POST-HOC PAIRWISE COMPARISONS")
    print(f"{'='*65}")

    # Find the best algorithm (lowest average rank)
    best_idx = np.argmin(avg_ranks)
    best_alg = algorithms[best_idx]
    print(f"Control algorithm (lowest avg rank): {best_alg} ({avg_ranks[best_idx]:.2f})")

    # Compute z-scores for each comparison against the control
    # z = (R_i - R_0) / sqrt(k(k+1) / 6N)
    k = n_algorithms
    N = n_instances
    se = np.sqrt(k * (k + 1) / (6 * N))

    comparisons = []
    for i in range(n_algorithms):
        if i != best_idx:
            z = (avg_ranks[i] - avg_ranks[best_idx]) / se
            # Two-sided p-value from normal distribution
            from scipy.stats import norm
            p = 2 * (1 - norm.cdf(abs(z)))
            comparisons.append((algorithms[i], avg_ranks[i], z, p))

    # Sort by p-value (ascending)
    comparisons.sort(key=lambda x: x[3])

    # Apply Holm correction
    m = len(comparisons)  # number of comparisons
    print(f"\n{'Comparison':<15} {'Avg Rank':>9} {'z-stat':>8} {'p-value':>10} "
          f"{'Holm α':>8} {'Result':>12}")
    print("-" * 65)

    for j, (alg, rank, z, p) in enumerate(comparisons):
        holm_alpha = 0.05 / (m - j)  # Holm correction
        significant = "SIGNIFICANT" if p < holm_alpha else "not signif."
        print(f"{best_alg} vs {alg:<6} {rank:>9.2f} {z:>8.4f} {p:>10.6f} "
              f"{holm_alpha:>8.4f} {significant:>12}")

    # ============================================================
    # Summary table for the report (Slide 16 format)
    # ============================================================
    print(f"\n{'='*65}")
    print(f"SUMMARY FOR REPORT (Slide 16 format)")
    print(f"{'='*65}")
    print(f"{'Algorithm':<6} {'Avg Rank':>10} {'CD test':>15}")
    print("-" * 35)

    for i, alg in enumerate(algorithms):
        if i == best_idx:
            print(f"{alg:<6} {avg_ranks[i]:>10.2f} {'best':>15}")
        else:
            # Find this algorithm in comparisons
            for comp_alg, _, _, p in comparisons:
                if comp_alg == alg:
                    # Check against its Holm-corrected alpha
                    for j, (ca, _, _, cp) in enumerate(comparisons):
                        if ca == alg:
                            holm_a = 0.05 / (m - j)
                            label = "diff." if cp < holm_a else "n.s."
                            print(f"{alg:<6} {avg_ranks[i]:>10.2f} {label:>15}")
                    break

print(f"\n{'='*65}")
print("Done. Include these results in Section 7.3 of your report.")
