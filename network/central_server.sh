#! /bin/bash

cd ../
source venv/bin/activate
cd network/

python new_server.py --ip_from_sensors $1 --port_from_sensors $2 --ip_to_sensors $3 --port_to_sensors $4
