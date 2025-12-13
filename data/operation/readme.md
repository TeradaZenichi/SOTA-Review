# Search strategy for operation-focused EV+BESS coordination studies

This document defines the search strategy for rebuilding the **operation** corpus, focusing on works that explicitly describe **near real-time control or dispatch** of electric vehicle charging together with battery energy storage. Compared to the planning scope, the emphasis here is on **short-timestep operation**, not siting or investment decisions.

We rely on conceptual blocks that capture the physical context (EV charging + BESS), the fact that the study is about operation or dispatch, and the specific control families we care about (MPC, rule-based, supervised ML, reinforcement learning). Two auxiliary blocks help push the query towards coordination and away from planning-only papers:

- **Coordination block**: `(coordinat* OR "joint optimization" OR co-optim* OR "integrated" OR "multi-objective" OR "coupled")`
- **Exclusion/planning block**: `(planning OR siting OR placement OR sizing OR "capacity planning" OR expansion OR investment OR layout OR "infrastructure planning")`

---

## 1. Core term blocks

### 1.1 EV charging block

```text
("electric vehicle" OR EV OR PEV OR BEV OR "plug-in") AND (charg* OR "smart charging" OR "managed charging" OR "charging scheduling" OR EVSE OR "charging station" OR "charging infrastructure")
```

### 1.2 BESS block

```text
("battery energy storage" OR BESS OR ESS OR "energy storage system" OR "stationary battery" OR "grid battery")
```

### 1.3 Operation/control block

```text
("real-time" OR realtime OR online OR "rolling horizon" OR "receding horizon" OR dispatch OR "energy management" OR control OR scheduling OR "operational")
```

### 1.4 Method blocks (we target four families)

- **Model Predictive Control**: `("model predictive control" OR MPC OR "receding horizon control" OR "rolling horizon")`
- **Rule-based / heuristic**: `("rule-based" OR heuristic* OR "priority-based" OR "threshold-based" OR "decision tree" OR "if-then" OR "state machine")`
- **Supervised Machine Learning for operation**: `("supervised learning" OR "machine learning" OR neural network OR LSTM OR "random forest" OR regression) AND (control OR scheduling OR dispatch OR policy)`
- **Reinforcement Learning / DRL**: `("reinforcement learning" OR "deep reinforcement learning" OR RL OR DRL OR DQN OR DDPG OR PPO OR SAC OR "actor-critic")`

### 1.5 Coordination block (optional but recommended)

```text
(coordinat* OR "joint optimization" OR co-optim* OR "integrated" OR "multi-objective" OR "coupled")
```

This block ensures the paper talks about **joint decision-making** (EV+BESS) rather than isolated optimization of one device.

### 1.6 Planning exclusions (use as `NOT (...)`)

```text
(planning OR siting OR placement OR sizing OR "capacity planning" OR expansion OR investment OR layout OR "infrastructure planning")
```

Apply this exclusion to avoid literature focused on long-term investment/siting.

---

## 2. Generic query template

```text
( EV block )
AND ( BESS block )
AND ( Operation block )
AND ( Method block for the desired control family )
AND ( Coordination block )
NOT ( Planning exclusions )
```

Note: the coordination block can be relaxed if the initial hitlist is already tight on operation.

---

## 3. Database-specific queries

### 3.1 Scopus (TITLE-ABS-KEY)

Use the following template, replacing the method block with the desired family (MPC/rule-based/supervised/RL). Example for MPC:

```text
TITLE-ABS-KEY(
  ("electric vehicle" OR EV OR PEV OR BEV OR "plug-in") AND (charg* OR "smart charging" OR "managed charging" OR "charging scheduling")
  AND ("battery energy storage" OR BESS OR ESS OR "energy storage system")
  AND ("real-time" OR realtime OR online OR dispatch OR "energy management" OR scheduling)
  AND ("model predictive control" OR MPC OR "rolling horizon")
  AND (coordinat* OR "joint optimization" OR "multi-objective" OR "integrated")
)
AND NOT TITLE-ABS-KEY(planning OR siting OR placement OR sizing OR "capacity planning" OR expansion OR investment OR layout OR "infrastructure planning")
```

Adapt the method-specific clause (lines 5) to `rule-based` heuristics, supervised ML or RL as needed.

### 3.2 Web of Science (Topic = TS)

Example for Reinforcement Learning:

```text
TS=(
  ("electric vehicle" OR EV OR PEV OR BEV OR "plug-in")
  AND (charg* OR "smart charging" OR "managed charging" OR "charging scheduling")
  AND ("battery energy storage" OR BESS OR ESS OR "energy storage system")
  AND ("real-time" OR online OR dispatch OR scheduling)
  AND ("reinforcement learning" OR "deep reinforcement learning" OR RL OR DRL OR DQN OR DDPG OR PPO)
  AND (coordinat* OR "joint optimization" OR "multi-objective" OR "coupled")
)
AND NOT TS=(planning OR siting OR placement OR sizing OR "capacity planning" OR expansion OR investment OR layout OR "infrastructure planning")
AND PY=2015-2025
AND DT=(Article)
AND WC=(ENGINEERING ELECTRICAL ELECTRONIC OR ENERGY FUELS)
```

Adjust the method clause for MPC/rule-based/supervised ML accordingly.

### 3.3 IEEE Xplore (metadata)

Use the metadata search with the following structure (example for supervised ML):

```text
("electric vehicle" OR EV OR PEV OR BEV)
AND (charg* OR "smart charging" OR "managed charging" OR "charging scheduling")
AND ("battery energy storage" OR BESS OR ESS)
AND ("real-time" OR online OR dispatch OR scheduling)
AND (("supervised learning" OR "machine learning" OR neural network OR LSTM OR "random forest") AND (control OR dispatch OR scheduling))
AND (coordinat* OR "joint optimization" OR "multi-objective")
AND NOT (planning OR siting OR placement OR sizing OR "capacity planning" OR expansion OR investment OR layout OR "infrastructure planning")
```

Apply filters: content type = Journals, years 2015–2025, subject area = Power & Energy / Transportation.

### 3.4 ScienceDirect (Title, abstract, keywords)

ScienceDirect limits boolean complexity, so run separate queries per method family. Example query for MPC with coordination:

```text
("electric vehicle" OR EV)
AND "battery energy storage"
AND ("real-time" OR dispatch OR scheduling)
AND ("model predictive control" OR MPC)
AND (coordinat* OR "joint optimization")
AND NOT (planning OR siting OR placement)
```

For rule-based, replace the MPC clause with `("rule-based" OR heuristic* OR "priority-based")`. For supervised ML, use `("supervised learning" OR "machine learning" OR neural network) AND (control OR dispatch)`. For RL, use `("reinforcement learning" OR DRL OR DQN OR PPO)`. Each query should be executed separately and exports stored under `data/operation/`.

---

## 4. Reproducibility notes

For each database, record:

- Exact query string and date executed.
# Search plan and exported queries (operation — EV + BESS)

I prepared and organized the search queries and export locations for reconstructing the operation corpus (focus: short-timestep control/dispatch of EV charging coordinated with BESS). Below I explain what I prepared and how to reproduce the searches.

What I prepared
- I created explicit queries for four method families: MPC, rule-based, supervised ML, and reinforcement learning.
- For sources that restrict boolean complexity (ScienceDirect) I prepared short/strict variants; for Scopus/WoS/IEEE I provided both full and short variants.
- I placed query files under `data/operation/` (one `query.txt` per source) and added a central reference with the method blocks.

How the queries are organized
- Each source file includes the full expressive query and a shorter variant where applicable. Run each variant separately if you need to compare recall vs precision.

Quick usage example
- Open `data/operation/sciencedirect/queries.txt` and copy the single-line query you prefer (COMPACT or STRICT).
- Run the search in ScienceDirect (Title/Abstract/Keywords), apply year filter 2015–2025 and export results into `data/operation/sciencedirect/` with a descriptive filename (e.g., `sciencedirect_mpc_2025-12-13.csv`).

Reproducibility notes
- Record the exact query string, date executed and any filters used for each export (file path and filename).
- After you have exported CSVs from all sources, run `scripts/operation/1-merge_raw_operation.py` to merge the files and continue the pipeline.

If you want, I can write four separate files for ScienceDirect (`query_mpc.txt`, `query_rule.txt`, `query_supervised.txt`, `query_rl.txt`) using either the compact or strict variants — tell me which variant and I will create them.

---

Relevant files in the repository
- `data/operation/scopus/query.txt` — Scopus queries (full + short)
- `data/operation/web-of-science/query.txt` — Web of Science queries (full + short)
- `data/operation/ieee/query.txt` — IEEE Xplore queries (full + short)
- `data/operation/sciencedirect/queries.txt` — ScienceDirect short queries (compact + strict)

Update this README with execution dates and exported filenames after you complete the searches.
