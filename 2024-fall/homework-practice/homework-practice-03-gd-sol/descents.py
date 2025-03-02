from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = -1  # w_0 -> w_1 is the first step and self.iteration will be 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(
        self,
        dimension: int,
        lambda_: float = 1e-3,
        loss_function: LossFunction = LossFunction.MSE,
    ):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        if self.loss_function != LossFunction.MSE:
            raise NotImplementedError("Need to implement loss function")
        return np.square(y - x.dot(self.w)).mean()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x.dot(self.w)


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        dw = np.negative(self.lr() * gradient)
        self.w += dw
        return dw

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function != LossFunction.MSE:
            raise NotImplementedError("Need to implement loss function")
        return 2 * x.T.dot(x.dot(self.w) - y) / len(y)


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(
        self,
        dimension: int,
        lambda_: float = 1e-3,
        batch_size: int = 50,
        loss_function: LossFunction = LossFunction.MSE,
    ):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sample = np.random.randint(len(y), size=self.batch_size)
        return super().calc_gradient(x[sample], y[sample])


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(
        self,
        dimension: int,
        lambda_: float = 1e-3,
        loss_function: LossFunction = LossFunction.MSE,
    ):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9
        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.alpha * self.h + self.lr() * gradient
        dw = np.negative(self.h)
        self.w += dw
        return dw


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(
        self,
        dimension: int,
        lambda_: float = 1e-3,
        loss_function: LossFunction = LossFunction.MSE,
    ):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * np.square(gradient)
        lr = self.lr()
        mhat = self.m / (1 - np.pow(self.beta_1, 1 + self.lr.iteration))
        vhat = self.v / (1 - np.pow(self.beta_2, 1 + self.lr.iteration))
        dw = np.negative(np.divide(lr, np.add(np.sqrt(vhat), self.eps)) * mhat)
        self.w += dw
        return dw


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = np.zeros_like(
            x.shape[1]
        )  # TODO: replace with L2 gradient calculation

        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get("descent_name", "full")
    regularized = descent_config.get("regularized", False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        "full": (
            VanillaGradientDescent if not regularized else VanillaGradientDescentReg
        ),
        "stochastic": StochasticDescent if not regularized else StochasticDescentReg,
        "momentum": MomentumDescent if not regularized else MomentumDescentReg,
        "adam": Adam if not regularized else AdamReg,
    }

    if descent_name not in descent_mapping:
        raise ValueError(
            f"Incorrect descent name, use one of these: {descent_mapping.keys()}"
        )

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get("kwargs", {}))
