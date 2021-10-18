# split classes 
# split = 0.02 ---------------> two classes (0.02 * 100)

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

# adam private data 10 classes each edge (selection method)
# split=0.1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace selection_private_data_3_workers_res18_adam_lambda_1_from_edge_ckpts \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --cloud_epoch 1000 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --two \
#                 --lamb 1 \
#                 --exist_loader \
#                 --selection \
#                 --resume \
#                 --alternate


# adam private data 5 classes each edge (selection method)
# split=0.05
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace selection_private_data_5_cls_single_edge_adam_lambda_1_again \
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
#                 --lamb 1 \
#                 --resume \
#                 --selection 

# adam private data 5 classes each edge two workers (selection method)
# split=0.02
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace full_data_2_cls_adam_lambda_1 \
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
#                 --lamb 1 \
#                 --two \
#                 --alternate \
#                 --selection 

########################################
# adam public data 5 classes each edge two workers (selection method)
# split=0.02
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace public_percent_0.5_2_cls_adam_lambda_1_private_distill \
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
#                 --lamb 1 \
#                 --two \
#                 --resume \
#                 --alternate \
#                 --public_distill \
#                 --public_percent 0.5 \
#                 --selection 

########################################
# adam public data 2 classes each edge two workers iid (selection method)

#public_percent_0.5_4_cls_adam_lambda_1_iid_public_distill_average

# split=0.03
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace ten_workers_iid_three_cls \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --num_workers 10 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --iid \
#                 --two \
#                 --alternate \
#                 --public_distill \
#                 --public_percent 0.5
                # --selection

# iid 
# split=0.04
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace seven_workers_iid_four_cls_res6 \
#                 --net res6 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --num_workers 7 \
#                 --num_rounds 1 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --iid \
#                 --two \
#                 --alternate \
#                 --public_distill \
#                 --finetune \
#                 --public_percent 0.5
#                 # --selection
#                 # --add_cifar10 \

# non-iid
# --workspace non_iid_public_res6_five_workers_5_cls \

# --workspace non_iid_public_res6_five_workers_2_cls_public_distill_25 \

split=0.02
CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
                --workspace non_iid_public_res8_five_workers_2_cls_public_distill_similarity \
                --net res8 \
                --cloud res18 \
                --split $split \
                --split_classes \
                --dataset cifar100 \
                --num_workers 5 \
                --num_rounds 1 \
                --epoch 200 \
                --cloud_epoch 200 \
                --optimizer adam \
                --cloud_lr 0.001 \
                --lr 0.1 \
                --lamb 1 \
                --two \
                --alternate \
                --public_distill \
                --public_percent 0.5
                --selection
                # --add_cifar10 \

# split=0.06
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace five_workers_iid_six_cls_res6_add_finetune \
#                 --net res6 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --num_workers 5 \
#                 --num_rounds 1 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --iid \
#                 --two \
#                 --alternate \
#                 --public_distill \
#                 --finetune \
#                 --public_percent 0.5
                # --selection
                # --add_cifar10 \




# adam private data (selection method, pseudo labels)
# split=0.1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace test_pl \
#                 --net res8 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 1 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --two \
#                 --lamb 1 \
#                 --exist_loader \
#                 --selection \
#                 --resume \
#                 --use_pseudo_labels \
#                 --alternate

# adam public data (selection method)
# split=0.1
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
#                 --workspace selection_public_data_3_workers_res18_adam_lambda_1 \
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
#                 --selection \
#                 --public_distill \
#                 --alternate

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
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace debug_load_ckpts \
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
#                 --resume \
#                 --selection \
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
#                 --workspace edge_train_private_test_public \
#                 --net res8 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 2 \
#                 --baseline \
#                 --lr 0.1 
                # --two \
                # --alternate \

# IID baseline case
# iid_cloud_baseline_use_first_again
# split=0.1
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py \
#                 --workspace iid_debug \
#                 --net res8 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --epoch 200 \
#                 --iid \
#                 --baseline \
#                 --lr 0.01
                # --two \
                # --alternate \