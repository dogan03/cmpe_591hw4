import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from homework5 import CNP
from utils import collectTrajectories


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


def test_cnp(cnp, X, Y, n_tests=100):
    cnp.eval()
    N = X.shape[0]
    ee_mse_list = []
    obj_mse_list = []

    for test_idx in range(n_tests):

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
            mean, std = cnp.forward(observation, target_x)
            mean = mean.squeeze(0)

            ee_mse = torch.mean(
                (mean[:, 0:2] - target_y.squeeze(0)[:, 0:2]) ** 2
            ).item()
            obj_mse = torch.mean(
                (mean[:, 2:4] - target_y.squeeze(0)[:, 2:4]) ** 2
            ).item()

            ee_mse_list.append(ee_mse)
            obj_mse_list.append(obj_mse)

        if (test_idx + 1) % 10 == 0:
            print(f"Completed {test_idx + 1}/{n_tests} tests")

    ee_mse_mean = np.mean(ee_mse_list)
    ee_mse_std = np.std(ee_mse_list)
    obj_mse_mean = np.mean(obj_mse_list)
    obj_mse_std = np.std(obj_mse_list)

    print("\nTest Results:")
    print(f"End-effector MSE: mean={ee_mse_mean:.4f}, std={ee_mse_std:.4f}")
    print(f"Object MSE: mean={obj_mse_mean:.4f}, std={obj_mse_std:.4f}")

    return ee_mse_mean, ee_mse_std, obj_mse_mean, obj_mse_std


def plot_results(ee_mse_mean, ee_mse_std, obj_mse_mean, obj_mse_std):
    labels = ["End-effector", "Object"]
    means = [ee_mse_mean, obj_mse_mean]
    stds = [ee_mse_std, obj_mse_std]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    width = 0.5

    bars = ax.bar(
        x,
        means,
        width,
        yerr=stds,
        align="center",
        alpha=0.7,
        ecolor="black",
        capsize=10,
    )

    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Prediction Error by Component")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + stds[i] + 0.001,
            f"Mean: {means[i]:.4f}\nStd: {stds[i]:.4f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    plt.tight_layout()
    plt.savefig("mse_results.png")
    print("Plot saved as 'mse_results.png'")
    plt.show()


if __name__ == "__main__":

    if not os.path.exists("cnp_model.pt"):
        print("Error: Model file 'cnp_model.pt' not found. Please run train.py first.")
        exit(1)

    test_data_path = "test_states_arr.npz"
    test_trajectories = 100

    print(f"Collecting {test_trajectories} new trajectories for testing...")
    collectTrajectories(
        n_trajectories=test_trajectories,
        render_mode="offscreen",
        trajectory_file_path=test_data_path,
    )

    print(f"Loading test data from {test_data_path}")
    data = np.load(test_data_path)
    data = [data[key] for key in data]

    X, Y = prepare_dataset(data)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    cnp = CNP(in_shape=[2, 4], hidden_size=128, num_hidden_layers=3)
    cnp.load_state_dict(torch.load("cnp_model.pt"))
    print("Model loaded successfully.")

    print(f"Running {100} tests...")
    ee_mse_mean, ee_mse_std, obj_mse_mean, obj_mse_std = test_cnp(
        cnp, X, Y, n_tests=100
    )

    plot_results(ee_mse_mean, ee_mse_std, obj_mse_mean, obj_mse_std)
