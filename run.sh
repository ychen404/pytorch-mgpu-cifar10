# split classes 
# split = 0.02 ---------------> two classes (0.02 * 100)

# The selection method
# split=0.02
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace test_selection \
#                 --net res20 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode uniform \
#                 --alpha 1 \
#                 --dataset cifar100 \
#                 --num_workers 2 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --selection \
#                 --public_distill \
#                 --public_percent 0.5 \

# the emb method
# split=0.02
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace test_emb \
#                 --net res20_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode uniform \
#                 --alpha 1 \
#                 --dataset cifar100 \
#                 --num_workers 2 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --public_distill \
#                 --public_percent 0.5 \
#                 --resume

# split=0.02
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace test_emb_res18 \
#                 --net res18_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode uniform \
#                 --alpha 1 \
#                 --dataset cifar100 \
#                 --num_workers 2 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --public_distill \
#                 --public_percent 0.5 \


# the emb method
# split=0.02
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace test_emb_res18_diri_1 \
#                 --net res18_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 1 \
#                 --dataset cifar100 \
#                 --num_workers 2 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --public_distill \
#                 --public_percent 0.5

# split=0.2
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace test_emb_diri_confidence \
#                 --net res18_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.1 \
#                 --dataset cifar10 \
#                 --num_workers 2 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --public_distill \
#                 --public_percent 0.5

# test to train the first edge model for 50 epoch, the seocnd for 1 epoch
# try to understand if norm of second to the last layer make sense

# split=1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_e_50_c_50 \
#                 --net res18_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 1 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 30 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --public_distill \
#                 --dlc \
#                 --public_percent 0.5

split=1
CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
                --workspace emb_diri_dlc_e_50_c_50_4cls \
                --net res6_emb \
                --cloud res18 \
                --split $split \
                --split_classes \
                --partition_mode dirichlet \
                --alpha 1 \
                --dataset cifar10 \
                --num_workers 3 \
                --num_rounds 50 \
                --epoch 50 \
                --cloud_epoch 50 \
                --optimizer adam \
                --cloud_lr 0.001 \
                --lr 0.1 \
                --lamb 1 \
                --temp 1 \
                --public_distill \
                --dlc \
                --public_percent 0.5


###### Test drop least confidence on cifar100 (sample 10 classes only) to compare with fedavg 
##### Not completed yet
# split=0.1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace test_emb_diri_drop_leastconfident \
#                 --net res20_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.1 \
#                 --dataset cifar10 \
#                 --num_workers 2 \
#                 --num_rounds 1 \
#                 --epoch 2 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --public_distill \
#                 --dlc \
#                 --public_percent 0.5