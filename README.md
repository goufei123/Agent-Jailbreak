# Test-driven Jailbreak Induction in Code Execution Agents

This repository provides the official implementation of the paper  
**“Test-driven Jailbreak Induction in Code Execution Agents”**.

The project studies how test-case generation workflows can be exploited to induce unintended or unsafe behaviors in LLM-based code execution agents.

---

## 🔍 Overview

Recent LLM-based code agents can autonomously write and execute programs inside sandboxed environments.  
However, their **test-case generation and debugging workflows** may unintentionally create opportunities for jailbreak-like behaviors.

This project proposes a **test-driven jailbreak induction framework**, which:

- Constructs realistic execution scenarios
- Prompts agents to generate and run test harnesses
- Observes whether unsafe or unintended behaviors occur
- Applies rollback strategies to iteratively induce execution

The framework enables systematic evaluation of **agent robustness and safety mechanisms**.

---

## 🧩 Pipeline Structure

The evaluation pipeline contains two main modes:

### 1. Direct Code Execution (RA)
The agent receives a code snippet and decides whether to:

- Execute it
- Modify it
- Reject it
- Finish the task

This evaluates baseline safety behavior.

### 2. Test-Case-Driven Execution (RA_TEST)
The agent is prompted to:

1. Generate a test harness
2. Execute the harness
3. Observe outcomes
4. Apply rollback strategies if rejected or failed

Four test-case generation modes are used:

- **Validation test**
- **Crash reproduction test**
- **Unit test**
- **Regression test**

If the agent rejects or fails:

- A **semantic rollback** modifies narrative framing
- An **execution rollback** strengthens test procedures

This iterative process increases the likelihood of inducing unsafe execution.

---

## 📁 Directory Structure

```bash
.
├── README.md
├── dataset/
│   └── RedCode-Exec/              # Python and Bash task datasets
│       ├── py2text_dataset_json/
│       └── bash2text_dataset_json/
│
├── environment/                   # Docker sandbox environment
│   ├── Dockerfile
│   ├── init.sh
│   ├── start_processes.sh
│   └── sensitive system files
│
├── evaluation/
│   └── RedCode_Exec/
│       ├── main.py                # Entry point for evaluation
│       ├── evaluation.py          # Scoring logic
│       ├── RA_evaluation/         # Direct execution pipeline
│       └── RA_evaluation_test/    # Test-driven jailbreak pipeline
│
├── scripts/
│   ├── RA_eval.sh
│   ├── RA_TEST_eval.sh
│   └── README.md
│
└── environment.yml                # Python environment configuration
