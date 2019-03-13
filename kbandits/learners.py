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

        # Greedy action
        greedy_action = self.learnedQ.argmax()
        action = greedy_action
        
        # Pick a non-greedy action
        if p <= self.eps:
            possible_actions = np.random.random_integers(0, 9, 2)
            action = (
                possible_actions[-1]
                if possible_actions[-1] != greedy_action
                else possible_actions[-2]
            )

        self.countActionChosen[action] += 1
        return action

    def give_reward(self, action, reward):
        diff = reward - self.learnedQ[action]
        self.learnedQ[action] += diff / self.countActionChosen[action]

    def reset(self):
        self.countActionChosen = np.zeros(10)
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


class UpperConfidenceBoundLearner(Learner):
    def __init__(self, eps, c):
        if eps < 0 or eps > 1:
            raise Exception("Invalid eps probability")

        self.eps = eps
        self.c = c
        # TODO: Maybe don't use the EpsGreedyClass
        self.learner = EpsGreedyLearner(eps)
        self.steps = 1

    def make_move(self):
        p = np.random.uniform()

        # Greedy action
        greedy_action = self.learner.learnedQ.argmax()
        action = greedy_action

        if p <= self.eps:

            # Default to any action that hasn't already been selected
            if (
                np.count_nonzero(self.learner.countActionChosen)
                != self.learner.countActionChosen.size
            ):
                action = self.learner.countActionChosen.argmin()
            else:
                # Calculate the UCB for each action
                ucb = self.learner.learnedQ + self.c * np.sqrt(
                    np.log(self.steps) / self.learner.countActionChosen
                )

                # Pick the max non-greedy action
                possible_actions = np.argpartition(ucb, -2)
                action = (
                    possible_actions[-1]
                    if possible_actions[-1] != greedy_action
                    else possible_actions[-2]
                )

        self.learner.countActionChosen[action] += 1
        return action

    def give_reward(self, action, reward):
        self.learner.give_reward(action, reward)

    def reset(self):
        self.learner.reset()
