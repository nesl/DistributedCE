source ../../venv/bin/activate
python load.py --video_files $1 --camera_ids $4 --yolo-weights ../soartechdetector2.engine --start_port $7 --server_port $8 --result_dir $9 --current_take ${10} ${11} --buffer_zone ${12} ${13} ${14}
