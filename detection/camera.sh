source ../../venv/bin/activate
python load.py --video_files $1 $2 $3 --camera_ids $4 $5 $6 --yolo-weights ../best_exp11.pt --start_port $7 --server_port $8 --result_dir $9 --current_take ${10} ${11} --buffer_zone ${12} ${13} ${14}
