import os

import numpy as np
import torch

from homework5 import CNP

task = "train"
data = np.load("states_arr.npz")
data = [data[key] for key in data]


def prepare_dataset(data):
    # data: list of (timesteps, 5) arrays: [e_y, e_z, o_y, o_z, h]
    all_inputs = []
    all_targets = []
    for traj in data:
        T = traj.shape[0]
        t = np.arange(T).reshape(-1, 1) / T  # normalized time
        h = traj[0, 4]  # object height is constant for trajectory
        h_arr = np.full((T, 1), h)
        x = np.concatenate([t, h_arr], axis=1)  # (T, 2): (t, h)
        y = traj[:, :4]  # (T, 4): (e_y, e_z, o_y, o_z)
        all_inputs.append(x)
        all_targets.append(y)
    X = np.concatenate(all_inputs, axis=0)
    Y = np.concatenate(all_targets, axis=0)
    return X, Y


import random


def evaluate_cnp(cnp, X, Y, n_tests=100):
    cnp.eval()
    N = X.shape[0]
    ee_mse_list = []
    obj_mse_list = []
    for _ in range(n_tests):
        # Randomly select context and target indices
        idx = np.arange(N)
        np.random.shuffle(idx)
        n_context = random.randint(1, N // 2)
        n_target = random.randint(1, N - n_context)
        context_idx = idx[:n_context]
        target_idx = idx[n_context : n_context + n_target]

        obs_x = X[context_idx]
        obs_y = Y[context_idx]
        tgt_x = X[target_idx]
        tgt_y = Y[target_idx]

        observation = torch.cat([obs_x, obs_y], dim=1).unsqueeze(0)  # (1, n_context, 6)
        target_x = tgt_x.unsqueeze(0)  # (1, n_target, 2)
        target_y = tgt_y.unsqueeze(0)  # (1, n_target, 4)

        with torch.no_grad():
            mean, _ = cnp.forward(observation, target_x)  # mean: (1, n_target, 4)
            mean = mean.squeeze(0)
            # End-effector: [:, 0:2], Object: [:, 2:4]
            ee_mse = torch.mean(
                (mean[:, 0:2] - target_y.squeeze(0)[:, 0:2]) ** 2
            ).item()
            obj_mse = torch.mean(
                (mean[:, 2:4] - target_y.squeeze(0)[:, 2:4]) ** 2
            ).item()
            ee_mse_list.append(ee_mse)
            obj_mse_list.append(obj_mse)
    print(
        f"End-effector MSE: mean={np.mean(ee_mse_list):.4f}, std={np.std(ee_mse_list):.4f}"
    )
    print(
        f"Object MSE: mean={np.mean(obj_mse_list):.4f}, std={np.std(obj_mse_list):.4f}"
    )


if __name__ == "__main__":
    if task == "train":
        X, Y = prepare_dataset(data)
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)

        cnp = CNP(in_shape=[2, 4], hidden_size=128, num_hidden_layers=3)
        optimizer = torch.optim.Adam(cnp.parameters(), lr=1e-3)
        cnp.train()

        for epoch in range(10000):
            # For simplicity, use all data as both context and target
            observation = torch.cat([X, Y], dim=1).unsqueeze(0)  # (1, N, 6)
            target_x = X.unsqueeze(0)  # (1, N, 2)
            target_y = Y.unsqueeze(0)  # (1, N, 4)
            loss = cnp.nll_loss(observation, target_x, target_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            if (epoch + 1) % 50 == 0:
                evaluate_cnp(cnp, X, Y, n_tests=100)
