sudo sysctl -w net.core.rmem_max=2147483647
sudo sysctl -w net.core.wmem_max=2147483647

python scripts/nerfcapture2dataset.py --save_path "/home/jay/4DTrack/data/iPhone_Captures/demo_1" --n_frames 1
