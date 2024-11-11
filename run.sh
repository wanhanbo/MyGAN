#!/bin/bash

# 获取当前时间，格式为 YYYYMMDD_HHMMSS
current_time=$(date +"%Y%m%d_%H%M%S")

# 定义日志文件名称
log_file="../../datasets/rock/output/log_$current_time.txt"

out_dir="../../datasets/rock/output/$current_time"

python3 skeletonRecallWGan_multi.py --out_dir $out_dir > $log_file