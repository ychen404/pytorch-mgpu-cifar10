#!/bin/bash
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.1 --workspace res6_2cls_10_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.2 --workspace res6_2cls_20_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.4 --workspace res6_2cls_40_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.6 --workspace res6_2cls_60_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.8 --workspace res6_2cls_80_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 1 --workspace res6_2cls_100_percent

# CUDA_VISIBLE_DEVICES=1 python3 single_model_test.py --net res34 --percent_classes 0.10 --percent_data 1 --workspace res34_10cls_100_percent
# python3 single_model_test.py --net res18 --percent_classes 0.2 --percent_data 0.5 --workspace res18_20cls_50_percent_save
python3 single_model_test.py --net res18 --percent_classes 0.02 --percent_data 0.5 --workspace res18_total_2cls_50pctdata

# python3 single_model_test.py --net res18 --percent_classes 0.10 --percent_data 0.25 --workspace res18_10cls_25_percent