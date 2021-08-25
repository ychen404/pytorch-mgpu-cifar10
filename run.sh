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
# concat_data_3_workers_res18_adam_lambda_0_baseline_lambda_0


# adam
split=0.1
CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
                --workspace public_data_distill_3_workers_lambda_0.5 \
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
                --public_distill \
                --alternate

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
# split=0.3
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace iid_cloud_baseline \
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