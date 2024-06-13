# Autonomous Vehicle Trajectory Planning

## Project Overview

This project focuses on implementing trajectory generation for an autonomous vehicle (ego vehicle) using various approaches, including the State Lattice Path Planner and Physics-based models. The goal is to enhance the accuracy and feasibility of trajectory predictions in dynamic and complex traffic scenarios.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Ensure you have the necessary dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```
## Configuration
Before running the training process, set the path to your data in the config.yaml file. Update the data_path field with the appropriate directory where your input data is stored.

```yaml
data_path: /path/to/your/data
```
## Running the Training
To train the model and generate the planned paths, run the following command:

```bash
python train.py
```
During the training process, the planned paths will be added to the input dictionary in the processing part.

## Visualization
The training process includes a visualization step that generates images of the scenarios. The visualizations will show:

The State Lattice Path Planner path in pink.
The Physics Model path in yellow.
These visualizations will help in qualitatively assessing the performance of different trajectory generation methods.

