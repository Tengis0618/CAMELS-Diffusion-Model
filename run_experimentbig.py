import os

learning_rates = [1e-5]
epochs_list = [100, 150, 200]
timesteps_list = [1500]

for lr in learning_rates:
    for epochs in epochs_list:
        for timesteps in timesteps_list:
            print(f"Running with lr={lr}, epochs={epochs}, timesteps={timesteps}")
            os.system(f"python big.py {lr} {epochs} {timesteps}")