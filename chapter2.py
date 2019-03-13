#%%
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [15, 5]

#%%
from kbandits import learners, valuedistributions, testrunner

#%% [markdown]
# ### Stationary Value Tests
# The following tests pit the epsilon-greedy algorithm against a stationary value distribution.

#%%
data_eps_1 = testrunner.run_test(
    learners.EpsGreedyLearner(0.1), valuedistributions.StationaryValue()
)
data_eps_01 = testrunner.run_test(
    learners.EpsGreedyLearner(0.01), valuedistributions.StationaryValue()
)
data_greedy = testrunner.run_test(
    learners.EpsGreedyLearner(0.0), valuedistributions.StationaryValue()
)

#%%
testrunner.plot_test_run_results(
    [data_eps_1, data_eps_01, data_greedy], ["eps=0.1", "eps=0.01", "eps=0.0 (greedy)"]
)


#%% [markdown]
# ### Non-stationary Value Tests
# Now we see how the epsilon-greedy algorithm works when the underyling value
# distribution is non-stationary.


#%%
data_non_stationary_sample_average = testrunner.run_test(
    learners.EpsGreedyLearner(0.1),
    valuedistributions.NonStationaryValue(),
    steps_per_run=5000,
)
data_non_stationary_constant_step = testrunner.run_test(
    learners.EpsGreedyConstantStepSizeLearner(eps=0.1, alpha=0.1),
    valuedistributions.NonStationaryValue(),
    steps_per_run=5000,
)


#%%
testrunner.plot_test_run_results(
    [data_non_stationary_sample_average, data_non_stationary_constant_step],
    ["Sample Average", "Constant Step Size"],
)

#%% [markdown]
# ### Optimistic Initial Values
# A greedy algo with optimisitic initial value estimates.


#%%
data_eps_realistic = testrunner.run_test(
    learners.EpsGreedyLearner(0.1), valuedistributions.StationaryValue()
)
data_greedy_optimistic = testrunner.run_test(
    learners.OptimisticEpsilonGreedyLearner(0.0, bias=5),
    valuedistributions.StationaryValue(),
)

#%%
testrunner.plot_test_run_results(
    [data_eps_realistic, data_greedy_optimistic], ["Eps Realistic", "Greedy Optimistic"]
)

#%% [markdown]
# ### Upper-Confidence-Bound Action Selection
# A eps-greedy algorithm that choses the non-greedy action with the maximal confidence-bound

#%%
data_eps_greedy_non_ucb = testrunner.run_test(
    learners.EpsGreedyLearner(0.1),
    valuedistributions.StationaryValue()
)
data_eps_greedy_ucb = testrunner.run_test(
    learners.UpperConfidenceBoundLearner(0.1, 2),
    valuedistributions.StationaryValue()
)

#%%
testrunner.plot_test_run_results(
    [data_eps_greedy_non_ucb, data_eps_greedy_ucb], ["Eps-Greedy eps=0.1", "UCB c=2"]
)


#%%
