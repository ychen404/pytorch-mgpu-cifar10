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
# public data to distill
# split=0.02
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace non_iid_public_res8_ten_workers_2_cls_private_distill_distill_pct_0.4 \
#                 --net res6 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --num_workers 10 \
#                 --num_rounds 1 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --two \
#                 --alternate \
#                 --public_distill \
#                 --public_percent 0.5 \
#                 --distill_percent 0.4 \
#                 --selection
                # --add_cifar10 \

# non-iid
# mixed private and public data to distill 
# split=0.02
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace non_iid_public_res8_10_workers_2_cls_mixed_distill_public_data_0.1pcnt_private \
#                 --net res6 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --num_workers 10 \
#                 --num_rounds 1 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --two \
#                 --alternate \
#                 --public_distill \
#                 --public_percent 0.5 \
#                 --private_mixed_percent 0.1 \
#                 --selection
#                 # --add_cifar10 \

# random edge for distillation
split=0.02
# non_iid_public_res6_10_workers_2_cls_0.1pcnt_private_random_edge
CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
                --workspace non_iid_public_res6_10_workers_2_cls_random_edge_mixed_10pcnt_private \
                --net res6 \
                --cloud res18 \
                --split $split \
                --split_classes \
                --dataset cifar100 \
                --num_workers 10 \
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
                --public_percent 0.5 \
                --private_mixed_percent 0.1 
                # --selection 
                # --random_edge
                # --add_cifar10 \

# non-iid
# private data
# split=0.02
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace non_iid_public_res8_ten_workers_2_cls_distill_fulldata \
#                 --net res6 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --num_workers 10 \
#                 --num_rounds 1 \
#                 --epoch 200 \
#                 --cloud_epoch 200 \
#                 --optimizer adam \
#                 --cloud_lr 0.001 \
#                 --lr 0.1 \
#                 --lamb 1 \
#                 --two \
#                 --alternate \
#                 --selection
                # --add_cifar10 \

# IID
# split=0.1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace iid_10_worker_res6_public_distill_unlabeled_lambda_1_50_pcnt_distill_alter_data_each_edge_50_truelabel \
#                 --net res6 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --num_workers 10 \
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
#                 --public_percent 0.5 
#                 # --distill_percent 0.5 \

# IID with random sampling (edge workers may have overlapped data)
# split=0.1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace iid_10_worker_res6_public_distill_unlabeled_lambda_1_50_pcnt_distill_alter_data_each_edge_random_sample_010 \
#                 --net res6 \
#                 --cloud res18 \
#                 --split $split \
#                 --split_classes \
#                 --dataset cifar100 \
#                 --num_workers 10 \
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
#                 --public_percent 0.5 \
#                 --random_sample \
#                 --sample_percent 0.10 
                # --distill_percent 0.5 \


# iid with finetune cloud model
# --workspace iid_5_workers_res6_2_cls_public_distill_finetune_50eps_all_publicdata \
# split=0.1
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
#                 --workspace iid_5_workers_res6_2_cls_public_distill_finetune_alllayers_50eps_all_publicdata \
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
#                 --public_percent 0.5 \
#                 --finetune \
#                 --finetune_epoch 50 \
#                 --finetune_percent 1