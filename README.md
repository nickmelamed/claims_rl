# ClaimsRL: Reinforcement Learning for Claim Verification

## Overview

**ClaimsRL** is a reinforcement learning (RL) framework for training agents to perform **iterative claim verification**. The system models claim evaluation as a sequential decision-making problem, where an agent interacts with an environment to gather evidence, construct arguments, and arrive at a final credibility judgment.

This project extends traditional LLM-based evaluation pipelines by introducing:

* **Action-based reasoning (select/remove/generate/finalize)**
* **Reward-driven learning (RLHF-style reward modeling)**
* **Debate-style reasoning (support vs. contradiction)**
* **Uncertainty-aware evaluation**

The goal is to demonstrate a deep integration of **LLMs + RL + evaluation systems** in a modular, extensible architecture.

For the inspiration behind the claims evaluation, check out this project: https://github.com/nickmelamed/co_claims

---

## Key Features

### RL Environment for Claim Verification

* Custom environment (`ClaimEnv`) simulates a claim verification workflow
* State includes:

  * Claim
  * Evidence pool
  * Selected evidence
  * Generated arguments
* Action space:

  * `SELECT`: choose evidence
  * `REMOVE`: discard evidence
  * `SUPPORT`: generate supporting argument
  * `CONTRADICT`: generate contradicting argument
  * `FINALIZE`: produce final decision

---

### Reward Model (RLHF-Inspired)

* Replaces traditional LLM judge with a **reward function**
* Provides:

  * Step-wise rewards (e.g., good evidence selection)
  * Final rewards (alignment with ground truth / expected reasoning)
* Designed to be:

  * Modular
  * Extensible to learned reward models later

---

### Multiple RL Strategies

Supports flexible experimentation with:

* **Multi-Armed Bandits**
* **Policy Gradient (PG)**
* **Proximal Policy Optimization (PPO)**

Switchable via a single configuration flag.

---

### Modular Policy Architecture

* `policy.act(state)` handles action selection
* Can be:

  * Random (baseline)
  * Learned (PG/PPO)
  * Hybrid (bandit + policy)

---

### Experiment Tracking

* Integrated `ExperimentTracker` for:

  * Logging rewards
  * Tracking episodes
  * Saving results to CSV
* Ensures reproducibility of runs

---

## Installation

```bash
git clone https://github.com/nickmelamed/claims_rl.git
cd claims_rl

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Project

### 1. Run a Single Episode

```bash
python run_episode.py
```

Outputs:

* Actions taken
* Intermediate rewards
* Final decision

---

### 2. Train the Agent

```bash
python train_rl.py
```

Configurable via `config.py`:

```python
RL_METHOD = "ppo"  # options: "bandit", "pg", "ppo"
NUM_EPISODES = 100
```

---

### 3. Track Results

Training automatically logs:

* Episode rewards
* Final scores
* Action distributions

Saved as:

```
logs/experiment_results.csv
```

---

## Reward Design

The reward function includes:

### Step Rewards

* Positive:

  * Selecting relevant evidence
  * Generating coherent arguments
* Negative:

  * Redundant or irrelevant actions
  * Excessive steps

### Final Reward

* Alignment with:

  * Expected outcome
  * Logical consistency
  * Evidence usage quality

---

## Example Workflow

1. Environment initialized with:

   * Claim: *"Company X increased revenue in 2024"*
   * Evidence set

2. Agent iteratively:

   * Selects evidence
   * Generates support/contradiction arguments

3. Agent calls `FINALIZE`

4. Reward model evaluates:

   * Evidence quality
   * Argument coherence
   * Final decision




