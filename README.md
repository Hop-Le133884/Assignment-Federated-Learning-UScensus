# Quickstart PyTorch — Federated Learning on UCI Adult Income

A federated learning example using **Flower** and **PyTorch** to train a binary classifier (income > $50K vs ≤ $50K) on the UCI Adult Income dataset across 5 simulated clients.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Step-by-Step Setup and Run](#step-by-step-setup-and-run)
  - [Step 1 — Clone or Download the Project](#step-1--clone-or-download-the-project)
  - [Step 2 — Install uv (Python Package Manager)](#step-2--install-uv-python-package-manager)
  - [Step 3 — Create the Root Virtual Environment](#step-3--create-the-root-virtual-environment)
  - [Step 4 — Preprocess the Data](#step-4--preprocess-the-data)
  - [Step 5 — Prepare Data Splits for Clients](#step-5--prepare-data-splits-for-clients)
  - [Step 6 — Generate Flower App Files and Workspace Environment](#step-6--generate-flower-app-files-and-workspace-environment)
  - [Step 7 — Run the Federated Learning Experiment](#step-7--run-the-federated-learning-experiment)
  - [Step 8 — Visualize the Results](#step-8--visualize-the-results)
- [Configuration](#configuration)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)

---

## What This Project Does

This project simulates a **federated learning** scenario:

- **5 clients** each hold a private, non-overlapping slice of training data (non-IID split by education level)
- A **central server** coordinates training over **10 communication rounds**
- Each round: clients train locally for 3 epochs → send model weights to server → server aggregates using **FedAvg** → sends updated global model back
- After each round, the server evaluates the global model on a held-out centralized test set
- Results (accuracy and loss per round) are saved as CSV files and can be visualized

**Model**: A 2-layer MLP: `87 features → 32 hidden (ReLU) → 1 output (raw logit)`

**Dataset**: [UCI Adult Income](https://archive.ics.uci.edu/dataset/2/adult) — 48,842 records, 14 demographic/economic features

---

## Prerequisites

Before starting, make sure you have the following installed on your system:

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.12+ | `python3 --version` |
| uv | latest | `uv --version` |
| Git | any | `git --version` |

> **Note**: A GPU is not required. The model is small and runs on CPU in a few minutes.

---

## Project Structure after the running the simulation

```
quickstart-pytorch/
├── data/
│   ├── adult/                          # Raw UCI Adult dataset
│   │   ├── adult.data                  # Training records (32,560 rows)
│   │   └── adult.test                  # Test records (16,281 rows)
│   └── preprocessed/                   # Auto-generated after Step 4
│       ├── adult_preprocessed.csv
│       ├── adult_test_preprocessed.csv
│       └── preprocessing_stats.json
├── utils/
│   ├── data_preprocessing.py           # Feature engineering & normalization
│   ├── data_preparation.py             # Non-IID data splitting for clients
│   └── jobs_gen.py                     # Generates Flower app files
├── workspace/                          # Auto-generated after Step 5 & 6
│   ├── client_1/ … client_5/           # Per-client training CSV splits
│   ├── server/test.csv                 # Centralized server test set
│   ├── model.py                        # MLP neural network
│   ├── task.py                         # Train / evaluate functions
│   ├── client_app.py                   # Flower ClientApp
│   ├── server_app.py                   # Flower ServerApp
│   ├── pyproject.toml                  # Workspace dependencies
│   └── report/                         # Auto-generated after Step 7
│       ├── accuracy_vs_rounds.csv
│       └── loss_vs_rounds.csv
├── preparing_data.ipynb                # Optional: interactive preprocessing notebook
├── acc_loss_visualization.py           # Plot accuracy & loss charts
├── data_splits_prep.sh                 # Shell script for Step 5
├── jobs_gen.sh                         # Shell script for Step 6
├── run_fl_experiment.sh                # Shell script for Step 7
└── pyproject.toml                      # Root project dependencies
```

---

## Step-by-Step Setup and Run

### Step 1 — Clone or Download the Project

```bash
git clone <repository-url>
cd Assignment-Federated-Learning-UScensus
```

Verify the raw dataset exists:

```bash
ls data/adult/
# Expected: adult.data  adult.names  adult.test  Index  old.adult.names
```

> If `data/adult/` is empty or missing, download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) and place `adult.data` and `adult.test` inside `data/adult/`.

---

### Step 2 — Install uv (Python Package Manager)

This project uses **uv** for fast, reproducible dependency management.

**Linux / macOS**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
```

After installation, reload your shell:
```bash
source ~/.bashrc   # or source ~/.zshrc
```

Verify:
```bash
uv --version
# e.g.: uv 0.7.x
```

---

### Step 3 — Create the Root Virtual Environment

From the project root, create and activate a virtual environment with Python 3.12:

```bash
uv venv --python 3.12
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install root dependencies (Flower, matplotlib, numpy, pandas):

```bash
uv sync
```

Verify the installation:
```bash
python -c "import flwr, pandas, matplotlib; print('All imports OK')"
```

---

### Step 4 — Preprocess the Data

This step cleans the raw UCI Adult dataset, normalizes numeric features, and one-hot encodes categorical features — producing 87 input features.

**Run Jupyter Notebook**:

Run all cells from top to bottom. The notebook walks through:
1. Exploring the raw data (shape, missing values, label distribution)
2. Preprocessing training data → `data/preprocessed/adult_preprocessed.csv`
3. Preprocessing test data → `data/preprocessed/adult_test_preprocessed.csv`

> **What preprocessing does**:
> - Normalizes: `age`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`
> - One-hot encodes: `workclass`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`
> - Label: `income` → 1 if `>50K`, else 0

---

### Step 5 — Prepare Data Splits for Clients

This step splits the preprocessed training data into **5 non-IID client datasets**, partitioned by education level.

```bash
./data_splits_prep.sh
```

The script accepts optional arguments:
```bash
./data_splits_prep.sh [NUM_CLIENTS] [WORKSPACE]
# Default: 5 clients, ./workspace
```

**What this creates**:

| Folder | Content | Education Levels |
|--------|---------|-----------------|
| `workspace/client_1/train.csv` | ~3,000 rows | Preschool – 5th grade |
| `workspace/client_2/train.csv` | ~2,500 rows | 6th – 8th grade |
| `workspace/client_3/train.csv` | ~9,000 rows | High school graduate |
| `workspace/client_4/train.csv` | ~7,000 rows | Some college |
| `workspace/client_5/train.csv` | ~8,000 rows | Bachelors – Doctorate |
| `workspace/server/test.csv` | ~16,000 rows | Centralized evaluation set |


### Step 6 — Generate Flower App Files and Workspace Environment

This step generates a bundle the Flower application code (`model.py`, `task.py`, `client_app.py`, `server_app.py`) 
into the workspace and sets up a **separate Python virtual environment** for running the simulation.

```bash
./jobs_gen.sh
```

Optional argument:
```bash
./jobs_gen.sh [WORKSPACE]
# Default: ./workspace
```

**What this creates**:
```
workspace/
├── model.py          # MLP definition
├── task.py           # load_data(), train(), evaluate()
├── client_app.py     # Flower ClientApp handlers
├── server_app.py     # Flower ServerApp + FedAvg aggregation
├── pyproject.toml    # Workspace-specific dependencies (torch, flwr)
└── .venv/            # Python 3.12 venv with workspace deps installed
```

This step also installs **PyTorch** and **Flower simulation** inside `workspace/.venv`


### Step 7 — Run the Federated Learning Experiment

Run the FL simulation. This executes 10 communication rounds, training the global model across 5 simulated clients.

```bash
./run_fl_experiment.sh
```

**What happens during the run**:

1. Flower spins up 5 virtual client nodes and 1 server node
2. **Round 0** (initialization): Server evaluates the untrained model on test set
3. **Rounds 1–10**: For each round:
   - Server sends global model weights to all 5 clients
   - Each client trains locally for 3 epochs on their data
   - Clients send updated weights back to server
   - Server aggregates via **FedAvg** (weighted average by number of samples)
   - Server evaluates global model on centralized test set
   - Metrics are logged to console and saved to CSV
4. Final model is saved to `workspace/final_model.pt`

**Expected console output** (abbreviated):
```
Running FL simulation → /path/to/workspace
=======================================
INFO : Starting Flower simulation...
INFO : [ROUND 1] global_accuracy=0.8347  global_loss=0.3542
INFO : [ROUND 2] global_accuracy=0.8469  global_loss=0.3286
...
INFO : [ROUND 10] global_accuracy=0.8503  global_loss=0.3238
INFO : Metrics saved to workspace/report/
```

**To stream logs for a specific run** (if run in background):
```bash
./run_fl_experiment.sh log <run-id>
```

After the run, verify results:
```bash
ls workspace/report/
# Expected: accuracy_vs_rounds.csv  loss_vs_rounds.csv

cat workspace/report/accuracy_vs_rounds.csv
```

---

### Step 8 — Visualize the Results

Generate two plots from the training metrics:
- **Bar chart** — Global accuracy and per-round evaluation accuracy across rounds
- **Line chart** — Global evaluation loss and training loss across rounds

Run from the `workspace/` directory:

```bash
cd workspace
python ../acc_loss_visualization.py
```

This reads:
- `report/accuracy_vs_rounds.csv`
- `report/loss_vs_rounds.csv`

And saves the output to:
- `report/acc_loss_visualization.png`

A window will also open with the plots displayed interactively.

---

## Configuration

The experiment is configured in `workspace/pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 10       # Number of FL communication rounds
fraction-evaluate = 0.6      # Fraction of clients used for evaluation per round
local-epochs = 1             # Local training epochs per client per round
learning-rate = 0.01        # Adam optimizer learning rate
batch-size = 64              # Mini-batch size for local training
```

To change the number of clients, pass an argument to the shell scripts:

```bash
./data_splits_prep.sh 10        # 10 clients
./jobs_gen.sh workspace_10      # into a different workspace
```

---

## Understanding the Output

| Metric | Round 0 | Round 10 | Meaning |
|--------|---------|---------|---------|
| `global_accuracy` | ~23% | ~85% | Accuracy of global model on server test set |
| `communication_round_eval_acc` | ~83% | ~85% | Average client local eval accuracy |
| `global_eval_loss` | ~0.75 | ~0.32 | Global model cross-entropy loss |
| `communication_round_train_loss` | ~0.43 | ~0.31 | Average client training loss |

> **Why does round 0 accuracy start at ~23%?** The model is randomly initialized. Round 0 runs a global evaluation before any training starts. The first real training round is round 1.

## Troubleshooting

**`./data_splits_prep.sh: Permission denied`**
```bash
chmod +x data_splits_prep.sh jobs_gen.sh run_fl_experiment.sh
```

**`ModuleNotFoundError: No module named 'flwr'`**
```bash
# Make sure the root venv is activated
source .venv/bin/activate
uv sync
```

**`Error: workspace venv not found. Run ./jobs_gen.sh first.`**
```bash
# Run jobs_gen.sh before run_fl_experiment.sh
./jobs_gen.sh
```

**`FileNotFoundError: data/preprocessed/adult_preprocessed.csv`**
```bash
# Run Step 4 first — data preprocessing
python utils/data_preprocessing.py \
    --input data/adult/adult.data \
    --output data/preprocessed/adult_preprocessed.csv \
    --stats data/preprocessed/preprocessing_stats.json
```

**`torch` not found when running visualization**
```bash
# acc_loss_visualization.py only needs pandas and matplotlib
# Make sure the root venv is active (not workspace venv)
source .venv/bin/activate
cd workspace
python ../acc_loss_visualization.py
```
