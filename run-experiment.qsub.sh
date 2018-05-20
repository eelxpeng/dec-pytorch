#!/bin/sh

#$ -S /bin/bash
#$ -V
#$ -cwd

#$ -w e
#$ -l h=client110

datasets="mnist stl reuters10k har"
for dataset in $datasets
do 
    python test_ae-3layer.py --dataset $dataset --save model/pretrained_"$dataset".pt
    python test_ltvae-3layer.py --dataset $dataset --lr 0.002 --epochs 20 --everyepochs 5 --pretrain model/pretrained_"$dataset".pt
done
# python test_ae-3layer.py --dataset mnist --lr 0.001 --epochs 100