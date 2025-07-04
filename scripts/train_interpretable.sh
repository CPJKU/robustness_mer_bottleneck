#!/bin/bash

for seed in 6 21 42 796 950 2025 2324 2451 3100 7192
do
   python -m train_models.train --arch a2m2e --task_loss_weight 0.5 --lr 0.0005 --seed ${seed}
done
