#!/bin/bash

# Initialize the arguments necessary
which_variable="Luminosity"
test_scenario="2CR"
sPOD_dim="1D"


mu_sets=(
  "0.001"
  "0.003"
  "0.005"
)

tau_sets=(
  "0.1"
  "0.2"
  "0.3"
  "0.4"
  "0.5"
  "0.6"
  "0.7"
  "0.8"
)

# TV regularization
gamma1="1e-3"
gamma2="1e-3"
gamma3="0"

# Modes upper limit
nmodes1="3"
nmodes2="3"
nmodes3="0"


for tau in "${tau_sets[@]}"; do
  for mu in "${mu_sets[@]}"; do
    echo "Submitting: Variable=$which_variable, Test_scenario=$test_scenario, sPOD_dim=$sPOD_dim, mu=$mu, tau=$tau"
    sbatch run_cluster_inner.sh "$which_variable" "$test_scenario" "$sPOD_dim" "$mu" "$tau" "$gamma1" "$gamma2" "$gamma3" "$nmodes1" "$nmodes2" "$nmodes3"
  done
done