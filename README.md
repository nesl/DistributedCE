# DistributedCE
Distributed Complex Event Detection Architecture

## Setup

1. Run `git clone --recurse-submodules https://github.com/nesl/DistributedCE.git`
2. Follow the instructions in [ComplexEventSimulator](https://github.com/nesl/ComplexEventSimulator) to install the simulator.
3. Create two virtual environments: **venv** using **requirements.txt** file in root directory, and **yolo** using **requirements.txt** file in **detection** directory (create it there).
4. Change directory: `cd ../network`
5. Clone [Mininet](https://github.com/mininet/mininet) and move the **mininet/mininet** subfolder to the current directory.

## Run

1. Run `sudo python3 test_mininet.py`

