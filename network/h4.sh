#! /bin/bash

cd ../
cd detection/
source yolo/bin/activate
python load.py --address 10.0.0.4 --port 58002
