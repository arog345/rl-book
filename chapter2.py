#%%
import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [15, 5]

#%%
import kbandits as kb

#%% [markdown]
# ### Stationary Value Tests
# The following tests pit the epsilon-greedy algorithm against a stationary value distribution.

#%%
data_eps_1 = kb.testrunner.run_test(
    kb.learners.EpsGreedyLearner(0.1), kb.valuedistributions.StationaryValue()
)
data_eps_01 = kb.testrunner.run_test(
    kb.learners.EpsGreedyLearner(0.01), kb.valuedistributions.StationaryValue()
)
data_greedy = kb.testrunner.run_test(
    kb.learners.EpsGreedyLearner(0.0), kb.valuedistributions.StationaryValue()
)

#%%
kb.testrunner.plot_test_run_results(
    [data_eps_1, data_eps_01, data_greedy], ["eps=0.1", "eps=0.01", "eps=0.0 (greedy)"]
)


#%% [markdown]
# ### Non-stationary Value Tests
# Now we see how the epsilon-greedy algorithm works when the underyling value
# distribution is non-stationary.


#%%
data_non_stationary_sample_average = kb.testrunner.run_test(
    kb.learners.EpsGreedyLearner(0.1),
    kb.valuedistributions.NonStationaryValue(),
    steps_per_run=5000,
)
data_non_stationary_constant_step = kb.testrunner.run_test(
    kb.learners.EpsGreedyConstantStepSizeLearner(eps=0.1, alpha=0.1),
    kb.valuedistributions.NonStationaryValue(),
    steps_per_run=5000,
)


#%%
kb.testrunner.plot_test_run_results(
    [data_non_stationary_sample_average, data_non_stationary_constant_step],
    ["Sample Average", "Constant Step Size"],
)

#%% [markdown]
# ### Optimistic Initial Values


#%%
data_eps_realistic = kb.testrunner.run_test(
    kb.learners.EpsGreedyLearner(0.1), kb.valuedistributions.StationaryValue()
)
data_greedy_optimistic = kb.testrunner.run_test(
    kb.learners.OptimisticEpsilonGreedyLearner(0.0, bias=5),
    kb.valuedistributions.StationaryValue(),
)


#%%
kb.testrunner.plot_test_run_results(
    [data_eps_realistic, data_greedy_optimistic], ["Eps Realistic", "Greedy Optimistic"]
)



#%%
