# Simulated Annealing for Job Shop Scheduling Problem (JSSP)

A Python implementation of Simulated Annealing (SA) algorithm for solving the Job Shop Scheduling Problem, with statistical comparison against other metaheuristic algorithms (Tabu Search, Genetic Algorithm).

## Overview

The Job Shop Scheduling Problem (JSSP) is a classic NP-hard combinatorial optimization problem. This project implements a high-performance Simulated Annealing algorithm with two parameter configurations (baseline and improved) and includes comprehensive statistical analysis for algorithm comparison.

### Key Features

- **Simulated Annealing Implementation**: Efficient SA solver for JSSP with configurable parameters
- **Dual Configurations**: Baseline and improved parameter sets for performance tuning
- **Automatic Temperature Calibration**: Initial temperature auto-calibrated per instance for optimal acceptance probability (~0.8)
- **Multi-Start Capability**: 5 independent runs per instance with different random seeds
- **Convergence Tracking**: Detailed logging of solution quality over time
- **Statistical Analysis**: Friedman test with Holm post-hoc analysis for algorithm comparison
- **Batch Processing**: Support for single instance or batch processing of multiple instances
- **Best Known Solutions (BKS)**: Built-in reference solutions from academic literature

## Requirements

- Python 3.7+
- `scipy` (for statistical tests)
- `numpy` (for numerical operations)

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd SimulatedAnnealing_JSSP
```

2. Install dependencies:
```bash
pip install scipy numpy
```

## Usage

### Running Simulated Annealing on JSSP Instances

#### Single Instance
```bash
python sa_jssp.py <instance_file>
```

#### Batch Processing (All instances in a folder)
```bash
python sa_jssp.py <folder_of_instances>
```

#### Using Improved Parameter Configuration
```bash
python sa_jssp.py <instance_file> --config improved
```

**Parameter configurations:**
- `baseline`: α=0.95, L=200, T_min=0.01 (standard settings)
- `improved`: α=0.97, L=500, T_min=0.01 (enhanced exploration)

### Statistical Comparison

Compare Simulated Annealing with Tabu Search and Genetic Algorithm:

```bash
python friedman_test.py
```

This performs a Friedman test followed by Holm post-hoc analysis across all 30 problem instances.

## Project Structure

```
.
├── sa_jssp.py              # Main SA implementation
├── friedman_test.py        # Statistical comparison framework
├── README.md               # This file
├── report/                 # Results and analysis reports
└── instances/              # JSSP benchmark instances
    ├── rcmax_*.txt         # Random instances (20 and 30 jobs)
    └── cscmax_*.txt        # Complex job correlation instances
```

### Benchmark Instances

The project includes 30 standard JSSP benchmark instances:

- **Set A**: 20 jobs, 15 machines (10 instances)
- **Set B**: 30 jobs, 15 machines (10 instances)  
- **Set C**: 30 jobs, 20 machines (10 instances)

## Algorithm Parameters

### Temperature Cooling Schedule
- **T₀**: Auto-calibrated per instance (target initial acceptance ≈ 0.8)
- **Cooling rate (α)**: 0.95 (baseline) or 0.97 (improved)
- **Minimum temperature (T_min)**: 0.01
- **Chain length (L)**: 200 (baseline) or 500 (improved)

### Neighborhood Operator
- Uses job swap perturbations in the permutation representation
- Evaluates makespan (total job completion time)

## Output

Results are saved in a timestamped directory with:
- **Summary CSV**: Overall performance metrics per instance
- **Convergence logs**: Detailed fitness progression for each run
- **Best solutions**: Optimal schedule found for each instance

## References

- **Best Known Solutions**: From [thomasWeise/jsspInstancesAndResults](https://github.com/thomasWeise/jsspInstancesAndResults) and BRKGA (Goncalves & Resende, 2013)
- **Statistical Methods**: Friedman test with Holm post-hoc analysis following methodology from Terci, 2026

## Project Information

- **Course**: CSE480/586 Term Project - Team 8
- **Algorithm**: Metaheuristic Optimization - Simulated Annealing
- **Problem Domain**: Combinatorial Optimization, Job Shop Scheduling