#!/bin/bash
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.1 --workspace res6_2cls_10_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.2 --workspace res6_2cls_20_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.4 --workspace res6_2cls_40_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.6 --workspace res6_2cls_60_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 0.8 --workspace res6_2cls_80_percent && \
# python3 single_model_test.py --net res6 --percent_classes 0.02 --percent_data 1 --workspace res6_2cls_100_percent

# CUDA_VISIBLE_DEVICES=1,2 python3 single_model_test.py --net res6 --dataset cifar10 --workspace res6_cifar10_full_data_200epochs_multistep --lr_sched multistep --epoch 200 --percent_data 1
# CUDA_VISIBLE_DEVICES=1,2 python3 single_model_test.py --net res8_aka --dataset cifar10 --workspace res8_aka_test --lr_sched None --epoch 1 --percent_data 1
# CUDA_VISIBLE_DEVICES=0,1,2 python3 single_model_test.py --net res6 --dataset cifar10 --workspace speedtest --lr_sched None --epoch 1 --percent_data 1

################ Cifar100 ################
# CUDA_VISIBLE_DEVICES=1 python3 single_model_test.py --net res34 --percent_classes 0.10 --percent_data 1 --workspace res34_10cls_100_percent
# python3 single_model_test.py --net res18 --percent_classes 0.2 --percent_data 0.5 --workspace res18_20cls_50_percent_save
# python3 single_model_test.py --net res18 --percent_classes 0.2 --lr 0.001 --percent_data 1 --workspace res18_20cls_100_percent_again_sgd_lr_0001

# python3 single_model_test.py --net res6 --percent_classes 0.10 --percent_data 0.25 --workspace res6_test_labels

################ ImageNet ################
# python3 single_model_test_imagenet.py --net res10 --percent_classes 1 --lr 0.001 --percent_data 1 --workspace res10_ign
# python3 single_model_test_imagenet.py --net res10 --percent_classes 1 --lr 0.01 --percent_data 1 --workspace res10_ign_alldata
# python3 single_model_test_imagenet.py --net res10 --dataset tiny --percent_classes 1 --lr 0.01 --percent_data 1 --workspace res10_tiny_ign_alldata

################ Downsampled ImageNet ################
# python3 single_model_test_downimagenet.py --net res6 --percent_classes 1 --lr 0.01 --percent_data 1 --workspace res6_downsampled_ign
# python3 single_model_test.py --net res6 --percent_classes 0.10 --dataset imagenet --epoch 200 --bs 128 --workspace res6_downsampled_ign_10pcnt_cls


################ Distillation test ################
CUDA_VISIBLE_DEVICES=0,1,2 python3 single_model_distill.py --net res6 --dataset cifar10 --workspace distill --cloud_epoch 1
