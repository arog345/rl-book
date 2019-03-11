from abc import ABC, abstractmethod
import numpy as np


class Learner(ABC):
    @abstractmethod
    def make_move(self):
        pass

    @abstractmethod
    def give_reward(self, selected_action, reward):
        pass

    @abstractmethod
    def reset(self):
        pass


class EpsGreedyLearner(Learner):
    def __init__(self, eps):
        if eps < 0 or eps > 1:
            raise Exception("Invalid eps probability")

        self.eps = eps

        self.countActionChosen = np.zeros(10)
        self.learnedQ = np.zeros(10)

    def make_move(self):
        p = np.random.uniform()

        action = 0

        if p <= self.eps:
            # Exploratory action
            action = np.random.randint(0, 10)
        else:
            # Greedy action
            action = self.learnedQ.argmax()

        self.countActionChosen[action] += 1
        return action

    def give_reward(self, action, reward):
        diff = reward - self.learnedQ[action]
        self.learnedQ[action] += diff / self.countActionChosen[action]

    def reset(self):
        self.countActionChosen = np.zeros(10)
        self.rewardsWhenActionChosen = np.zeros(10)
        self.learnedQ = np.zeros(10)


class EpsGreedyConstantStepSizeLearner(Learner):
    def __init__(self, eps, alpha):
        if eps < 0 or eps > 1:
            raise Exception("Invalid eps probability")

        self.eps = eps
        self.alpha = alpha

        self.learnedQ = np.zeros(10)

    def make_move(self):
        p = np.random.uniform()

        action = 0

        if p <= self.eps:
            # Exploratory action
            action = np.random.randint(0, 10)
        else:
            # Greedy action
            action = self.learnedQ.argmax()

        return action

    def give_reward(self, action, reward):
        self.learnedQ[action] += self.alpha * (reward - self.learnedQ[action])

    def reset(self):
        self.countActionChosen = np.zeros(10)
        self.rewardsWhenActionChosen = np.zeros(10)
        self.learnedQ = np.zeros(10)


class OptimisticEpsilonGreedyLearner(Learner):
    def __init__(self, eps, bias=0):
        self.bias = bias
        self.learner = EpsGreedyLearner(eps)
        self.learner.learnedQ.fill(bias)

    def make_move(self):
        return self.learner.make_move()

    def give_reward(self, selected_action, reward):
        self.learner.give_reward(selected_action, reward)

    def reset(self):
        self.learner.reset()
        self.learner.learnedQ.fill(self.bias)
