# This is repository for reproducing GAR on Reasoning Intensive IR



## Installation

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- CUDA-compatible GPU (for local model inference)
- Java 11+ (for PyTerrier)

### Setup


1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   
   ```bash
   pip install -r requirements.txt
   # Additional PyTerrier plugins
   pip install git+https://github.com/terrierteam/pyterrier_pisa.git
   pip install --upgrade git+https://github.com/emory-irlab/pyterrier_genrank.git
   pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git
   ```


## Preprocessing: Index and Graph Creation

Before running experiments, you may need to create an index and a corpus graph.

### 1. Create Index
Use `create_index.py` to index the dataset.

```bash
python3 create_index.py --task <TASK_NAME> 
```

**Arguments:**
- `--task`: The specific task from the BRIGHT benchmark (default: `biology`).

### 2. Create Graph
Use `create_graph.py` to create a corpus graph for adaptive reranking.

```bash
python3 create_graph.py --task <TASK_NAME> --k <K>
```

**Arguments:**
- `--task`: The specific task (default: `biology`).
- `--k`: Number of neighbors in the graph (default: `16`).


## Usage

The main entry point for running experiments is `run.py`.

### Running an Experiment

To run a reranking experiment on a specific task within the BRIGHT benchmark:

```bash
python3 run.py --task <TASK_NAME> --model_name <MODEL> --budget <TOP_K>
```


### Examples

Run Rank1-7B on the Robotics task with a budget of 100 documents:
```bash
python3 run.py --task robotics --model_name rank1-7b --budget 100
```

Run TFRank-8B on the Sustainable Living task:
```bash
python3 run.py --task sustainable-living --model_name tfrank-8b --budget 50
```

## Budget 50 results

Due to limited space, we added budget c=50 results per subtask in `budget_50.md`
