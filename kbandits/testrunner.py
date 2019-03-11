import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def perform_run(learner, value_distribution, num_steps=1000):
    reward = np.zeros(num_steps, dtype="float64")
    was_optimal = np.zeros(num_steps, dtype="float64")

    for i in range(num_steps):
        selected_action = learner.make_move()
        reward[i] = value_distribution.reward(selected_action)
        learner.give_reward(selected_action, reward[i])
        was_optimal[i] = (
            1 if selected_action == value_distribution.optimal_action() else 0
        )

    return reward, was_optimal


def run_test(
    learner, value_distribution, num_runs=2000, steps_per_run=1000, enable_logging=False
):
    avg_reward = np.zeros(steps_per_run, dtype="float64")
    percent_optimal = np.zeros(steps_per_run, dtype="float64")

    for i in range(num_runs):
        learner.reset()
        value_distribution.reset()

        run_reward, run_was_optimal = perform_run(
            learner, value_distribution, steps_per_run
        )

        avg_reward += run_reward
        percent_optimal += run_was_optimal

        if enable_logging and (i + 1) % 100 == 0:
            print(f"Completed run {i+1} of {num_runs}.")

    avg_reward /= num_runs
    percent_optimal /= num_runs

    test_data = pd.DataFrame(
        {"avg_reward": avg_reward, "percent_optimal": percent_optimal}
    )
    test_data.index.name = "step"

    return test_data


def plot_test_run_results(results, labels):
    colors = "brgcmykw"

    ax1 = plt.subplot(1, 2, 1)
    for i, (r, l) in enumerate(zip(results, labels)):
        plt.plot(r.index, r["avg_reward"], colors[i % len(colors)], label=l)
    plt.xlabel("Step")
    plt.ylabel("Average Reward")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    for i, (r, l) in enumerate(zip(results, labels)):
        plt.plot(r.index, r["percent_optimal"], colors[i % len(colors)], label=l)
    plt.xlabel("Step")
    plt.ylabel("% Optimal")
    ax2.yaxis.set_major_formatter(mp.ticker.PercentFormatter(1))
    ax2.legend()

    plt.show()
