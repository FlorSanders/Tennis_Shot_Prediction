# Predictive Modeling of Tennis Player Poses and Ball Trajectory (TECO)

## Introduction

The goal of the project is to perform time series prediction of player poses and ball trajectory in tennis matches.
In order to limit computational complexity, a feature extraction pipeline will be built using open source models for object detection and pose estimation, to convert dense video into a sparse 3D representation of the point.
These extracted features will serve to build a time series prediction model based on graph neural networks or transformer architectures.
The ability to predict the movement of players and ball in a tennis point unlocks many downstream applications, from analytics to coaching.

## Setup

Clone the repository to make a local copy and change directory.

```bash
git clone git@github.com:FlorSanders/adl_ai_tennis_coach.git
cd adl_ai_tennis_coach
```

Separate install instructions are provided in the modules of the `src` directory, as data processing, training and inference have different sets of required dependencies.
