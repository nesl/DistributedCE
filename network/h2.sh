#! /bin/bash

cd ../
cd detection/
source yolo/bin/activate
python load.py --address $1 --port $2 --address_to_server $3 --port_to_server $4 --address_from_server $5 --port_from_server $6 --camera_id $7
#python load.py --address 10.0.0.2 --port 58001
