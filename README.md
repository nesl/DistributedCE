# DistributedCE
Distributed Complex Event Detection Architecture

## Setup

1. Install CARLA 0.9.10 with the Additional Maps. Extract the CARLA files and put them under a new directory with the name of **CARLA_0.9.10**.
2. Install poetry into your system
3. Install Scenic by getting into the Scenic folder and using `poetry install -E dev`
4. Clone [CARLA-2DBBox](https://github.com/MukhlasAdib/CARLA-2DBBox) into the main directory and name it bbox_annotation.
5. Run the command `CARLA_0.9.10/CarlaUE4.sh` in one terminal tab.
6. Run the command `scenic pedestrian.scenic --simulate` in another tab.
