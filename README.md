# Predictive Modeling of Tennis Player Poses and Ball Trajectory (TECO)

## Introduction

The goal of the project is to perform time series prediction of player poses and ball trajectory in tennis matches.
In order to limit computational complexity, a feature extraction pipeline will be built using open source models for object detection and pose estimation, to convert dense video into a sparse 3D representation of the point.
These extracted features will serve to build a time series prediction model based on graph neural networks or transformer architectures.
The ability to predict the movement of players and ball in a tennis point unlocks many downstream applications, from analytics to coaching.

## Setup

In order to install the dependencies, initiate a conda environment from the `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate teco
```

If changes are made, the environment can be exported using the provided script.

```bash
bash export-environment.sh
```
