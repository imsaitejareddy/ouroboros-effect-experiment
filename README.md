# Ouroboros Effect Experiment

This repository contains the code, data, results, and reports for a set of experiments validating the Ouroboros Effect – a hypothesised feedback loop in which model collapse and human cognitive off‑loading mutually amplify degradation.

## Contents

- `rigorous_studies.py` – Python script that simulates the formal Ouroboros ODE model and runs machine‑learning experiments on the `digits` dataset. It generates JSON results and plots.
- `ode_results.json` and `ml_results.json` – Results of the ODE and machine‑learning simulations.
- `*.png` – Plots showing the evolution of model quality, human feedback quality, and test accuracy under different conditions.
- `report_high_value_studies.md` – A detailed report analysing the experiments, summarising methods, results, and implications.
- `ouroboros_study_package.zip` – Archive containing all files from this project.

## Running the experiment

To reproduce the experiments locally:

1. Install dependencies:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```
2. Run the simulation script:
   ```bash
   python rigorous_studies.py
   ```

The script will generate JSON result files and PNG plots in the current directory.

## Summary

The baseline simulation shows a coupled decline in model quality and human feedback quality when synthetic errors are recursively accepted. Provenance-aware retrieval (RAG) completely halts this decline, and an evolutionary algorithm slows but does not prevent degradation when used alone. Combining RAG with the evolutionary algorithm yields performance similar to RAG alone, suggesting that provenance is the most effective mitigation strategy.
