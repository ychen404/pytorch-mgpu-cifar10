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

############### distillation ###############


############### Drop-worst ###############

# split=0.4
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_r_1_e_50_c_100_4cls_alpha_0.01 \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_r_1_e_50_c_100_10cls_alpha_0.01 \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5


# split=1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_r_1_e_50_c_100_10cls_alpha_0.01_public_0.2 \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.2

# split=1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_r_1_e_50_c_100_10cls_alpha_100 \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_r_1_e_50_c_100_10cls_alpha_100_5workers \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_homo_r_100_e_10_c_3_10cls_alpha_100_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace check_confidence_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5 \
#                 --save_confidence

# split=0.2
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace check_confidence_res18_multistep_200ep_uniform_5workers \
#                 --net res18_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode uniform \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 200 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lr_sched multistep \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5 \
#                 --save_confidence

# split=0.2
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace check_confidence_uniform_200ep_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode uniform \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 200 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5 \
#                 --save_confidence

# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace check_confidence_test \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 1 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5 \
#                 --save_confidence

################ Weighted average based on confidence ################

# split=1
# CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
#                 --workspace emb_diri_wavg_new_homo_r_100_e_10_c_50_10cls_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 50 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_wavg_new_homo_r_100_e_10_c_3_10cls_alpha_0.1_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.1 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_wavg_r_1_e_50_c_100_10cls_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_wavg_r_100_e_100_c_3_10cls_alpha_100_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 100 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5


# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_wavg_adam_test_batch_r_100_e_10_c_3_10cls_alpha_100_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

split=1
CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
                --workspace debug \
                --net res6_emb \
                --cloud res6 \
                --split $split \
                --split_classes \
                --partition_mode dirichlet \
                --alpha 100 \
                --dataset cifar10 \
                --num_workers 5 \
                --num_rounds 100 \
                --epoch 1 \
                --cloud_epoch 3 \
                --optimizer adam \
                --cloud_lr 0.001 \
                --lr 0.1 \
                --lamb 1 \
                --temp 1 \
                --public_distill \
                --emb_mode wavg \
                --aggregation_mode distillation \
                --public_percent 0.5




##### Reduce cloud data
# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_wavg_reduce_cloud_data_adam_r_100_e_40_c_3_10cls_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 40 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --reduce_cloud_data \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_wavg_speedtest_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.1 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 1 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# #### No decay
# split=1
# CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
#                 --workspace emb_diri_wavg_nodecay_r_100_e_100_c_3_10cls_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 100 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --no_decay \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace debug \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 100 \
#                 --epoch 1 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --no_decay \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

############### fedavg ###############

# split=0.4
# CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_4cls_fedavg_alpha_0.01 \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 100 \
#                 --epoch 5 \
#                 --cloud_epoch 10 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode fedavg \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_10cls_fedavg_alpha_0.01 \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 100 \
#                 --epoch 5 \
#                 --cloud_epoch 10 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode fedavg \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_dlc_10cls_fedavg_alpha_100 \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 3 \
#                 --num_rounds 100 \
#                 --epoch 5 \
#                 --cloud_epoch 10 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode fedavg \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_10cls_fedavg_10ep_alpha_0.1_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.1 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 10 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode fedavg \
#                 --public_percent 0.5

## 25 rounds 20 local epochs
# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_10cls_fedavg_r_25_e_20_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 25 \
#                 --epoch 20 \
#                 --cloud_epoch 10 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode fedavg \
#                 --public_percent 0.5

# ------------- No decay --------------
# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_10cls_wavg_nodecay_10ep_alpha_100_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --no_decay \
#                 --public_distill \
#                 --emb_mode wavg \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# ------------- Use akamaster resnet -------------
# split=1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace emb_diri_akamaster_10cls_fedavg_alpha_100_5workers \
#                 --net res8_aka \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 5 \
#                 --cloud_epoch 10 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --aggregation_mode fedavg \
#                 --public_percent 0.5

############## FedDF ##############
# if drop no device, then the dlc method is equivalent to FedDF
# 10 local epochs

                # --workspace emb_diri_feddf_nodecay_cosineaneal_r_100_e_10_c_3_10cls_alpha_100_5workers \

# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_feddf_r_100_e_10_c_3_10cls_alpha_0.1_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.1 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --num_drop 0 \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

###### cosine annealing
# split=1
# CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
#                 --workspace emb_diri_feddf_cosineaneal_r_100_e_10_c_3_10cls_alpha_100_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --lr_sched cos \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --num_drop 0 \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

###### reduce distill data
# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_feddf_reduce_distill_data_r_100_e_100_c_1_10cls_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 100 \
#                 --cloud_epoch 1 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --reduce_cloud_data \
#                 --emb_mode dlc \
#                 --num_drop 0 \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5


# 50 local epochs, one-shot
# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_feddf_r_1_e_50_c_100_10cls_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode dlc \
#                 --num_drop 0 \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5


################ Fed-ET ################
# split=1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace emb_diri_fedet_r_1_e_50_c_100_10cls_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 50 \
#                 --cloud_epoch 100 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode fedet \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace emb_diri_fedet_diversity_reg_r_100_e_10_c_3_10cls_alpha_0.01_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.01 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode fedet \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace emb_diri_fedet_diversity_reg_r_100_e_10_c_3_10cls_alpha_100_5workers \
#                 --net res6_emb \
#                 --cloud res6 \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 100 \
#                 --dataset cifar10 \
#                 --num_workers 5 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode fedet \
#                 --aggregation_mode distillation \
#                 --public_percent 0.5

# split=1
# CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
#                 --workspace emb_diri_fedet_r_100_e_10_res8_c_3_vgg19_10cls_alpha_0.1_10workers \
#                 --net res8_emb \
#                 --cloud vgg \
#                 --split $split \
#                 --split_classes \
#                 --partition_mode dirichlet \
#                 --alpha 0.1 \
#                 --dataset cifar10 \
#                 --num_workers 10 \
#                 --num_rounds 100 \
#                 --epoch 10 \
#                 --cloud_epoch 3 \
#                 --cloud_batch_size 5 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --temp 1 \
#                 --public_distill \
#                 --emb_mode fedet \
#                 --aggregation_mode distillation \
#                 --public_percent 0.7