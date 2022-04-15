# split classes 
# split = 0.02 ---------------> two classes (0.02 * 100)

split=0.02
CUDA_VISIBLE_DEVICES=1 python3 train_cifar.py \
                --workspace non_iid_res8_ten_workers_2_cls_public_distill \
                --net res6 \
                --cloud res18 \
                --split $split \
                --split_classes \
                --dataset cifar100 \
                --num_workers 2 \
                --num_rounds 1 \
                --epoch 2 \
                --cloud_epoch 2 \
                --optimizer adam \
                --cloud_lr 0.001 \
                --lr 0.1 \
                --lamb 1 \
                --alternate \
                --public_distill \
                --public_percent 0.5 \
                --selection