# The Anti-Ouroboros Effect: Empirical Evidence for Resilience in AI Feedback Loops

This repository contains the code and results for a series of experiments investigating feedback loops in AI models. The research began by validating the "Ouroboros Effect" in a simple classifier and culminated in the discovery of the **"Anti-Ouroboros Effect"** in a large language model (LLM).

## 📂 Experiments

This repository contains two primary experiments:

### 1. Large Language Model (LLM) Summarization Experiment (Gemma 2B)

This is the main experiment detailed in the associated paper. It uses the `google/gemma-2b-it` model to test recursive feedback loops on a scientific summarization task (`ccdv/arxiv-summarization`) over five generations.

-   **Key Finding:** Contrary to the original hypothesis, a simple automated quality filter did not accelerate model collapse. Instead, it induced an "Anti-Ouroboros Effect," leading to a robust improvement in model performance (ROUGE-L score), while the unfiltered control arm showed degradation.
-   **Location:** [`/llm_gemma_experiment/`](./llm_gemma_experiment/)

### 2. Simple Classifier Experiment (Digits Dataset)

This initial study simulated the Ouroboros Effect using an ODE model and a simple classifier on the `digits` dataset.

-   **Key Finding:** This experiment successfully validated the original Ouroboros hypothesis under controlled conditions, showing a coupled decline in model and feedback quality.
-   **Location:** [`/simple_digits_experiment/`](./simple_digits_experiment/)

---

## 🚀 Running the Experiments

### LLM Summarization Experiment (Main)

The entire experiment can be replicated by running the Jupyter Notebook. This experiment was conducted on a **Tesla P100 GPU**.

1.  **Navigate to the experiment directory:**
    ```bash
    cd llm_gemma_experiment
    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the notebook:**
    Open `ouroboros_llm_gemma.ipynb` in a Jupyter environment and execute the cells. The notebook handles all data loading, model fine-tuning, and evaluation.

### Simple Classifier Experiment (Initial Study)

1.  **Navigate to the experiment directory:**
    ```bash
    cd simple_digits_experiment
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib scikit-learn
    ```
3.  **Run the simulation script:**
    ```bash
    python rigorous_studies.py
    ```

---

## 📈 Key Results

The primary result is the discovery of the **Anti-Ouroboros Effect** in the LLM experiment. The quality filter created a positive feedback loop, improving the model with each generation.

![LLM Performance Plot](./llm_gemma_experiment/results_plot.png)

| Generation | Control_QM | Ouroboros_QM |
|------------|------------|--------------|
| 0          | 0.1638     | 0.1638       |
| 1          | 0.1737     | 0.1641       |
| 2          | 0.1716     | 0.1658       |
| 3          | 0.1724     | 0.1681       |
| 4          | 0.1741     | 0.1716       |
| 5          | 0.1684     | 0.1746       |

*(Table data extracted from the notebook run)*
