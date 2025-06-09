import os

learning_rates = [1e-5, 1e-4]
epochs_list = [100, 150]
timesteps_list = [1000, 1500]

for lr in learning_rates:
    for epochs in epochs_list:
        for timesteps in timesteps_list:
            print(f"Running with lr={lr}, epochs={epochs}, timesteps={timesteps}")
            os.system(f"python initial2.py {lr} {epochs} {timesteps}")
