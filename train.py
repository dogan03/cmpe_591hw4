import os
import random

import numpy as np
import torch

from homework5 import CNP


def prepare_dataset(data):

    all_inputs = []
    all_targets = []
    for traj in data:
        T = traj.shape[0]
        t = np.arange(T).reshape(-1, 1) / T
        h = traj[0, 4]
        h_arr = np.full((T, 1), h)
        x = np.concatenate([t, h_arr], axis=1)
        y = traj[:, :4]
        all_inputs.append(x)
        all_targets.append(y)
    X = np.concatenate(all_inputs, axis=0)
    Y = np.concatenate(all_targets, axis=0)
    return X, Y


def evaluate_cnp(cnp, X, Y, n_tests=100):
    cnp.eval()
    N = X.shape[0]
    ee_mse_list = []
    obj_mse_list = []
    for _ in range(n_tests):

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

        observation = torch.cat([obs_x, obs_y], dim=1).unsqueeze(0)
        target_x = tgt_x.unsqueeze(0)
        target_y = tgt_y.unsqueeze(0)

        with torch.no_grad():
            mean, _ = cnp.forward(observation, target_x)
            mean = mean.squeeze(0)

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
    return (
        np.mean(ee_mse_list),
        np.std(ee_mse_list),
        np.mean(obj_mse_list),
        np.std(obj_mse_list),
    )


if __name__ == "__main__":
    data = np.load("states_arr.npz")
    data = [data[key] for key in data]
    X, Y = prepare_dataset(data)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    cnp = CNP(in_shape=[2, 4], hidden_size=128, num_hidden_layers=3)
    optimizer = torch.optim.Adam(cnp.parameters(), lr=1e-3)
    cnp.train()

    num_epochs = 500
    eval_interval = 50
    best_loss = float("inf")

    for epoch in range(num_epochs):
        observation = torch.cat([X, Y], dim=1).unsqueeze(0)
        target_x = X.unsqueeze(0)
        target_y = Y.unsqueeze(0)

        loss = cnp.nll_loss(observation, target_x, target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        if (epoch + 1) % eval_interval == 0:
            ee_mse_mean, _, obj_mse_mean, _ = evaluate_cnp(cnp, X, Y, n_tests=100)
            avg_mse = (ee_mse_mean + obj_mse_mean) / 2

            if avg_mse < best_loss:
                best_loss = avg_mse
                torch.save(cnp.state_dict(), "cnp_model.pt")
                print(f"Model saved at epoch {epoch+1} with avg MSE: {avg_mse:.4f}")

    print("Training completed. Best model saved as 'cnp_model.pt'")
