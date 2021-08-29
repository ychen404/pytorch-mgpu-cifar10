# split classes 
# split = 0.02 ---------------> two classes (0.02 * 100)

# split=0.1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace alternate_data_3_workers_res18_cloud_epoch_400 \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --cloud_epoch 400 \
#                 --lr 0.1 \
#                 --two \
#                 --alternate

# alternate_data_3_workers_res18_adam_lambda_0
# concat_data_3_workers_res18_adam_lambda_1
# data_selection_distill_3_workers_lambda_0.5
# data_selection_distill_3_workers_lambda_1

# adam private data
# split=0.1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace baseline_only_true_label_concat_data_3_workers_res18_adam_lambda_0 \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --two \
#                 --lamb 0 \
#                 --exist_loader \
#                 --alternate


# adam private data (selection method)
# split=0.1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace selection_private_data_3_workers_res18_adam_lambda_1 \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --two \
#                 --lamb 1 \
#                 --exist_loader \
#                 --selection \
#                 --alternate

# adam public data (selection method)
split=0.1
CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
                --workspace selection_public_data_3_workers_res18_adam_lambda_1 \
                --net res8 \
                --cloud res18 \
                --split $split \
                --split_classes \
                --dataset cifar100 \
                --epoch 200 \
                --cloud_epoch 200 \
                --optimizer adam \
                --cloud_lr 0.001 \
                --lr 0.1 \
                --two \
                --lamb 1 \
                --selection \
                --public_distill \
                --alternate

# adam public data
# split=0.1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace concat_data_3_workers_res18_adam_lambda_1_again \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --two \
#                 --lamb 1 \
#                 --public_distill \
#                 --alternate


# DEBUG
# split=0.1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace collect_edge_checkpoints \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --two \
#                 --lamb 1 \
#                 --exist_loader \
#                 --alternate


# SGD
# split=0.1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace check_two_class_alternate \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --lr 0.1 \
#                 --two \
#                 --exist_loader \
#                 --alternate

# Baseline template
# Also need to turn off the flags for exist loader and save loader
# split=0.1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace test_baseline \
#                 --net res20 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 1 \
#                 --baseline \
#                 --lr 0.1
                # --two \
                # --alternate \

# IID baseline case
# iid_cloud_baseline_use_first_again
# split=0.3
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace iid_cloud_baseline_debug \
#                 --net res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --iid \
#                 --baseline \
#                 --lr 0.1
#                 # --two \
#                 # --alternate \