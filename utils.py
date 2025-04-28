import os

import numpy as np

from homework5 import Hw5Env, bezier


def collectTrajectories(
    n_trajectories=100, render_mode="offscreen", trajectory_file_path="states_arr.npz"
):
    env = Hw5Env(render_mode=render_mode)
    if os.path.exists(trajectory_file_path):
        states_arr = np.load(trajectory_file_path)
        states_arr = [states_arr[key] for key in states_arr]
    else:
        states_arr = []
    try:
        for i in range(n_trajectories):
            env.reset()
            p_1 = np.array([0.5, 0.3, 1.04])
            p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
            p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
            p_4 = np.array([0.5, -0.3, 1.04])
            points = np.stack([p_1, p_2, p_3, p_4], axis=0)
            curve = bezier(points)

            env._set_ee_in_cartesian(
                curve[0],
                rotation=[-90, 0, 180],
                n_splits=100,
                max_iters=100,
                threshold=0.05,
            )
            states = []
            for p in curve:
                env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
                states.append(env.high_level_state())
            states = np.stack(states)
            states_arr.append(states)
            print(f"Collected {len(states_arr)} trajectories.", end="\r")
    finally:
        print("Stopping collection...")
        print("Saving trajectories with length: ", states_arr.__len__())
        np.savez(trajectory_file_path, *states_arr)


if __name__ == "__main__":
    collectTrajectories(n_trajectories=1000)
