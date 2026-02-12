#!/bin/bash

export PROJECT_PATH="/data/yimingwang/Interpreting_ToM/"
export CUDA_VISIBLE_DEVICES="0"

mode="train"
test="test"

layer_num=16
hidden_size=1024
head_num=24

n=4
m=5
x=1
y=3
k=1

infer_n=4
infer_m=5
infer_x=1
infer_y=3
infer_k=1


python pretrain.py --mode $mode \
                    --test $test \
                    --layer_num $layer_num \
                    --hidden_size $hidden_size \
                    --head_num $head_num \
                    --n $n \
                    --m $m \
                    --x $x \
                    --y $y \
                    --k $k \
                    --infer_n $infer_n \
                    --infer_m $infer_m \
                    --infer_x $infer_x \
                    --infer_y $infer_y \
                    --infer_k $infer_k