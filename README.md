# AutoCheck
### AUTOmated CHECKpointing System for Machine Learning Models and Long-Running High-Performance Computing Jobs

> Graduation Project — Faculty of Computers and Artificial Intelligence, Cairo University  
> Academic Year 2025–2026  
> Supervised by **Dr. Ahmed Shawky Moussa**

---

## Table of Contents

- [Abstract](#abstract)
- [Background & Motivation](#background--motivation)
- [Problem Definition](#problem-definition)
- [Features](#features)
- [Architecture](#architecture)
- [System Workflow](#system-workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Work Plan](#work-plan)
- [Team](#team)

---

## Abstract

High-performance computing (HPC) clusters impose strict execution time limits on jobs and are prone to hardware failures. Modern machine learning training tasks often require days, weeks, or even months to complete. When a job is terminated mid-execution, users are forced to restart from the beginning — creating a restart loop where long-running jobs may never finish.

**AutoCheck** is an automated checkpointing system that periodically saves the complete training state of machine learning models and HPC jobs **without modifying the training code**. It enables reliable job resumption across multiple time-limited sessions, significantly reducing wasted computation time and improving experiment completion rates on HPC systems.

---

## Background & Motivation

HPC clusters allow multiple users to share large compute resources (CPUs, GPUs). While powerful, they enforce strict per-job time limits for fair resource allocation. Deep learning models in particular can require training times spanning several weeks. When a job exceeds its time limit or encounters a failure, all unsaved progress is lost.

Manual checkpointing exists but is:
- Error-prone and requires code modification
- Not reliably triggered on unexpected interruptions
- Tightly coupled to specific training scripts, reducing portability

AutoCheck addresses all of these limitations with a fully autonomous, transparent, and model-agnostic approach.

---

## Problem Definition

The core problem is the **absence of an automated and reliable mechanism** that transparently captures the complete training state of machine learning jobs — including model parameters, optimizer state, and training progress — without requiring modifications to the training code.

Without such a system, researchers must choose between abandoning long-running experiments or risking repeated loss of progress.

---

## Features

- **Zero code modification** — works transparently alongside any training loop
- **Model-agnostic** — supports PyTorch and TensorFlow without changing the checkpointing engine
- **Comprehensive state snapshots** — saves model weights, optimizer state, scheduler state, hyperparameters, training metadata, and custom user-defined objects
- **Autonomous controller** — runs in a background thread, triggering checkpoints based on configurable time intervals or session limits
- **Structured storage** — organizes checkpoints in a hierarchical directory layout (`/checkpoints/stage/epoch-batch/`) with dedicated metadata files
- **Extensible registry** — register additional training components without modifying the core engine
- **YAML configuration** — all parameters controlled via a single config file

---

## Architecture

```
┌─────────────────────┐          ┌──────────────────────────┐
│   Configuration     │◄─────────│      Runner Script       │
│   Layer (YAML)      │          │  train_with_checkpoints()│
└─────────────────────┘          └────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌─────────────────────────┐
                                 │   Autonomous Controller  │
                                 │      (Orchestrator)      │
                                 └──────┬──────────┬────────┘
                                        │          │
                              ▼                    ▼
                  ┌──────────────────┐   ┌─────────────────────┐
                  │   State Tracker  │   │  Checkpoint Manager  │
                  │  epoch / batch   │   │    (save / load)     │
                  │  loss / accuracy │   └──────────┬──────────┘
                  └──────────────────┘              │
                                                    ▼
                                       ┌────────────────────────┐
                                       │    Persistence Layer   │
                                       │      (File System)     │
                                       └────────────────────────┘
```

The system is composed of loosely coupled components that operate **independently of the training code**:

| Component | Responsibility |
|---|---|
| **ConfigManager** | Loads and parses the YAML configuration file |
| **RunnerScript** | Entry point; starts training and initializes the checkpointing system |
| **AutonomousController** | Background thread; monitors state and triggers checkpoints based on policy |
| **StateTracker** | Shared object tracking epoch, batch, loss, accuracy, and elapsed time |
| **CheckpointManager** | Handles saving and loading of full training state (abstract base with PyTorch & TensorFlow implementations) |
| **Persistence Layer** | Structured file system storage for checkpoint files and metadata |

---

## System Workflow

1. The user starts a training job and points AutoCheck to a **configuration YAML file**
2. The training loop runs normally — **no checkpoint logic embedded**
3. The **Runner Script** updates the shared **State Tracker** with current epoch, batch, loss, and accuracy
4. The **Autonomous Controller** runs in a background thread, monitoring training state and elapsed time
5. Based on configurable policies (checkpoint interval or session time limit), the controller triggers the **Checkpoint Manager**
6. The Checkpoint Manager saves model parameters, optimizer state, metadata, and registered objects to a structured directory
7. On job termination (time limit or failure), a final checkpoint is saved automatically
8. On job resumption, the controller loads the most recent checkpoint and training continues from the last saved state

---

## Requirements

- Python 3.8+
- PyTorch or TensorFlow
- PyYAML

```bash
pip install torch pyyaml        # for PyTorch
pip install tensorflow pyyaml   # for TensorFlow
```

---

## Installation

```bash
git clone https://github.com/Maria-alfonse/AutoCheck.git
cd AutoCheck
pip install -r requirements.txt
```

---

## Configuration

All checkpointing behavior is controlled through a `config.yaml` file:

```yaml
checkpointing:
  enabled: true
  interval_minutes: 30        # Save a checkpoint every 30 minutes
  session_time_limit: 120     # Stop and save after 120 minutes (HPC time limit)
  storage_path: ./checkpoints
  framework: pytorch          # pytorch or tensorflow

training:
  epochs: 100
  batch_size: 64
```

---

## Usage

```python
from autocheck import RunnerScript

# Point to your model, dataloader, and config — no changes to your training loop
runner = RunnerScript(config_path="config.yaml")
runner.train_with_checkpoints(model, dataloader, optimizer)
```

To resume an interrupted job:

```bash
python run.py --resume --config config.yaml
```

AutoCheck automatically detects the latest checkpoint and resumes from there.

---

## Project Structure

```
AutoCheck/
├── config.yaml                  # Example configuration file
├── run.py                       # Entry point
├── autocheck/
│   ├── config_manager.py        # YAML loader and config accessor
│   ├── runner_script.py         # Training entry point with checkpoint hooks
│   ├── autonomous_controller.py # Background thread orchestrator
│   ├── state_tracker.py         # Shared training state object
│   ├── checkpoint_manager.py    # Abstract base checkpoint manager
│   ├── pytorch_manager.py       # PyTorch implementation
│   └── tensorflow_manager.py    # TensorFlow implementation
├── checkpoints/                 # Auto-generated checkpoint storage
│   └── stage/
│       └── epoch-batch/
├── requirements.txt
└── README.md
```

---

## Work Plan

| Task | Description | Status |
|---|---|---|
| Define idea & scope | Problem statement, feasibility, objectives | ✅ Completed |
| Research | Literature review, DMTCP & ML framework analysis | ✅ Completed |
| Documentation & presentation | Midyear report and slides | ✅ Completed |
| Config Layer | YAML parser and config module | 🔄 Feb 21, 2026 |
| State Tracker | Real-time job and training state monitoring | 🔄 Mar 1, 2026 |
| Checkpoint Manager | Save, load, and recovery logic | 🔄 Mar 13, 2026 |
| Autonomous Controller | Background orchestration and coordination | 🔄 Apr 18, 2026 |
| Runner Script | Job launcher with checkpoint integration | 🔄 Apr 26, 2026 |
| Testing & packaging | End-to-end tests and final release | 🔄 May 1, 2026 |

---

## Team

| Name |
|---|
| Toqa Abdalla Ahmed |
| Habiba Ayman Hamed |
| Kermina Nashaat Shafiek |
| Maria Alfons Kamel |

> Faculty of Computers and Artificial Intelligence, Cairo University  
> Department of Computer Science
