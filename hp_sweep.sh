#!/bin/bash

# Define hyperparameters
gammas=(0.0 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 1)
Ns=(1 2 5 10 25 50 100 250 500 1000)
sigma_inits=(0.01 0.025 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Default values
trials=1000
eta0=0.1
w_init=0.0
seed=0
nstates=10

max_jobs=12  # Limit concurrent jobs
job_count=0

for gamma in "${gammas[@]}"; do
    for N in "${Ns[@]}"; do
        for sigma_init in "${sigma_inits[@]}"; do
            ((job_count++))
            echo "Launching job $job_count: gamma=$gamma, N=$N, sigma_init=$sigma_init"
            python backward_shift_fields.py \
                --trials $trials \
                --gamma $gamma \
                --eta0 $eta0 \
                --N $N \
                --sigma_init $sigma_init \
                --w_init $w_init \
                --seed $seed \
                --nstates $nstates &
            
            # Limit the number of background jobs
            if (( job_count % max_jobs == 0 )); then
                wait  # Wait for current batch to finish
            fi
        done
    done
done

wait  # Wait for remaining jobs
echo "All experiments completed!"