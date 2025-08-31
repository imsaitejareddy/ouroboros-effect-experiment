# The Anti-Ouroboros Effect: Empirical Evidence for Resilience in AI Feedback Loops

This repository contains the official code, data, and results for the paper, "The Anti-Ouroboros Effect: Empirical Evidence for Resilience in AI Feedback Loops." The research began by validating the "Ouroboros Effect" in a simple classifier and culminated in the discovery of the **"Anti-Ouroboros Effect"** in a large language model (LLM).

## 📂 Experiments

This repository contains two primary experiments:

### 1. Large Language Model (LLM) Summarization Experiment (Gemma 2B)

This is the main experiment detailed in the paper. It uses the `google/gemma-2b-it` model to test recursive feedback loops on a scientific summarization task (`ccdv/arxiv-summarization`) over five generations.

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
(Instructions for the digits experiment follow here...)

---

## 🔁 Reproducibility

This project is committed to full reproducibility. All necessary components are publicly available:

* **Code:** All scripts and notebooks used for the experiments are provided in this repository.
* **Dataset:** The `ccdv/arxiv-summarization` dataset is publicly available on the Hugging Face Hub.
* **Model:** The base model, `google/gemma-2b-it`, is an open-weight model available on Hugging Face.
* **Environment:** The exact library versions are specified in the `requirements.txt` file within each experiment's folder. The main experiment was run using a Tesla P100-16GB GPU.

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

---

## 📜 Citation

If you use the code or findings from this research, please cite the paper:

```bibtex
@article{Adapala2025AntiOuroboros,
  author  = {Adapala, Sai Teja Reddy},
  title   = {The Ouroboros Effect and its Inverse: Empirical
Studies of AI Feedback Loop Dynamics},
}
```
