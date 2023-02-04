#! /bin/bash

cd ../
source venv/bin/activate
cd network/
python broker.py --port_consumers $1 --port_producers $2
