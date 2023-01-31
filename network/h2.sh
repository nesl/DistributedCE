#! /bin/bash

cd ../
cd detection/
source yolo/bin/activate
python load.py --address $1 --port $2
#python load.py --address 10.0.0.2 --port 58001
