# Neural Networks for Quantum-System Simulation

Materials for the 2026 CECAM School lecture series on *Neural Networks for Quantum-System Simulation*.

Welcome! This repository contains the lecture notes, tutorial notebooks, and supporting software used throughout the course.

The goal of these lectures is to provide an introduction to neural quantum states (NQS), a family of machine-learning methods for representing and simulating quantum many-body systems. We will combine ideas from quantum mechanics, variational methods, and modern machine learning to study how neural networks can be used to address the quantum many-body problem.

The course is designed to be hands-on. Alongside the lectures, you will find tutorial notebooks covering both minimal implementations from scratch and examples based on modern scientific machine-learning tools such as JAX and NetKet. The aim is not only to understand the underlying concepts, but also to gain practical experience with the software and workflows commonly used in contemporary NQS research. Hope you will enjoy it, see you soon!

## Repository structure

```text
notes/
    Lecture notes

notebooks/
    Tutorial notebooks

src/cecam_nqs/
    Shared code used throughout the tutorials
```

The notebooks are designed to accompany the lectures and should be completed in the order in which they appear.

## Getting started

Before attending the tutorials, all participants should run (please do read the next section before you do):

```text
notebooks/00_environment_check.ipynb
```

This notebook verifies that the required software is correctly installed and that JAX and NetKet are functioning as expected.

## Running the notebooks

### Option 1: Google Colab (recommended)

All notebooks can be run directly in Google Colab.

Open the notebook using the corresponding Colab badge and save a personal copy to your Google Drive before editing.

No local installation is required.

### Option 2: Local installation using uv

Install uv:

https://docs.astral.sh/uv/getting-started/installation/

Clone the repository:

```bash
git clone https://github.com/Z-Denis/cecam-nqs.git
cd cecam-nqs
```

Create the environment and install all dependencies:

```bash
uv sync
```

Launch JupyterLab:

```bash
uv run jupyter lab
```

Then open:

```text
notebooks/00_environment_check.ipynb
```

and verify that the installation works correctly before proceeding to the tutorials.

### Option 3: Existing Python/Jupyter environment

Advanced users may install the package manually in an existing Python environment.

However, the course material is developed and tested using the `uv` environment provided in this repository. Using the provided environment is strongly recommended.

## Updating the materials

New notebooks and supporting material will be released throughout the school.

Before each tutorial session, a tagged release of the repository will be created. This ensures that all participants are working from exactly the same version of the course material.

In normal use, participants do not need to interact with these tags directly.

If running locally, simply update the repository before each tutorial:

```bash
git pull
uv sync
```

Participants using Google Colab will automatically access the corresponding notebook version through the provided Colab links.

## Contact

For questions regarding the lecture material, please contact:

Zakari Denis

[zakari.denis@gmail.com](mailto:zakari.denis@gmail.com)

