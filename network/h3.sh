#! /bin/bash

cd ../
source venv/bin/activate
cd detection/
python forwarder.py --address_source $1 --address_sink $2 --port_source $3 --port_sink $4
# python forwarder.py --address_source 10.0.0.1 --address_sink 10.0.0.2 --port_source 55001 --port_sink 58001
