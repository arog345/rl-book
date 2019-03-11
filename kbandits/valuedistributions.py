from abc import ABC, abstractmethod
import numpy as np


class ValueDistribution(ABC):
    @abstractmethod
    def reward(self, action):
        pass

    @abstractmethod
    def optimal_action(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class StationaryValue(ValueDistribution):
    def __init__(self, size=10):
        self.size = size

        self.value = np.random.randn(size)
        self.optimal_action_val = self.value.argmax()

    def reward(self, action):
        return self.value[action] + np.random.randn()

    def optimal_action(self):
        return self.optimal_action_val

    def reset(self):
        self.value = np.random.randn(self.size)
        self.optimal_action_val = self.value.argmax()


class NonStationaryValue(ValueDistribution):
    def __init__(self, size=10, drift_mean=0.0, drift_std=0.01):
        self.drift_mean = drift_mean
        self.drift_std = drift_std

        self.value = np.random.randn(size)
        self.original_value = np.copy(self.value)

    def reward(self, action):
        self.value += np.random.normal(
            self.drift_mean, self.drift_std, self.value.shape
        )
        return self.value[action] + np.random.randn()

    def optimal_action(self):
        return self.value.argmax()

    def reset(self):
        self.value = np.copy(self.original_value)
