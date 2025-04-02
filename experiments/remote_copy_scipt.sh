#!/bin/bash

# Define the grid of latent dimensions and the seed.
latent_dim_grid=(2)
SEED=42
MODEL_TYPE="RVAE"

# Loop over the latent dimensions.
for latent_dim in "${latent_dim_grid[@]}"; do
    echo "Processing latent dimension d${latent_dim}"
    
    # Define the remote source checkpoint path and the local destination.
    remote_source="knumentropy:/home/knum/projekty/rvae_reproduction/saved_models/d${latent_dim}/${SEED}/${MODEL_TYPE}/"
    local_dest="/home/prz/PROJECTS/rvae_reproduction/saved_models/d${latent_dim}/${SEED}/"
    
    # Create the local destination folder if it doesn't exist.
    mkdir -p "$local_dest"
    
    # Copy the checkpoint
    scp -r "$remote_source" "$local_dest"
done

echo "All checkpoints have been copied."