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

# adam
# split=0.1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace alternate_data_3_workers_res18_adam \
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
#                 --alternate

# SGD
split=0.1
CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
                --workspace two_classes_alternate \
                --net res8 \
                --cloud res18 \
                --split $split \
                --split_classes \
                --dataset cifar100 \
                --epoch 2 \
                --cloud_epoch 2 \
                --lr 0.1 \
                --two \
                --exist_loader \
                --alternate


# Baseline template
# Also need to turn off the flags for exist loader and save loader
# split=0.1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace res20_30cls_baseline \
#                 --net res20 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --baseline \
#                 --lr 0.1
#                 # --two \
#                 # --alternate \