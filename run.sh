# split classes 
# split = 0.02 ---------------> two classes (0.02 * 100)

split=0.1

#CUDA_VISIBLE_DEVICES=2 python3 train_cifar.py --workspace cifar100_$split\_data_$split\_classes_res50 --net res50 --split $split --split_classes --dataset cifar100 

# not split classes
CUDA_VISIBLE_DEVICES=3 python3 train_cifar.py \
                --workspace alternate_data \
                --net res18 \
                --split $split \
                --split_classes \
                --dataset cifar100 \
                --epoch 2 \
                --two \
                --lr 0.1