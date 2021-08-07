# split classes 
# split = 0.02 ---------------> two classes (0.02 * 100)

split=0.1

#CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py --workspace cifar100_$split\_data_$split\_classes_res50 --net res50 --split $split --split_classes --dataset cifar100 

# not split classes
CUDA_VISIBLE_DEVICES=0 python3 train_cifar.py \
                --workspace cifar100_swap_five_classes_lambda_1 \
                --net res8 \
                --split $split \
                --split_classes \
                --dataset cifar100 \
                --epoch 5 \
                --two \
                --lr 0.001
