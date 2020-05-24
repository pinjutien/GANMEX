#!/usr/bin/env bash

for run in 'A' 'B'
do
    for w in '10' '20' '50' '100' '200'
    do
        python -m tensorflow_gan.examples.cyclegan.train -cycle_consistency_loss_weight ${w}.0 -train_log_dir /tmp/tfgan_logdir_h2z_grid_w${w}_${run}/cyclegan/ -max_number_of_steps 10000 -save_checkpoint_steps 2000
    done
done
