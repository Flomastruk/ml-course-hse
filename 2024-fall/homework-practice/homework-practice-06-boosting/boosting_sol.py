from __future__ import annotations

from collections import defaultdict

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from plotly.subplots import make_subplots
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

layout_dict = dict(
    margin=dict(l=20, r=20, t=40, b=20),
    width=600,
    height=400,
    paper_bgcolor="LightSteelBlue",
    title_font_size=14,
    xaxis_title_font_size=12,
    yaxis_title_font_size=12,
)


def score(clf: Boosting, x, y, n=None):
    return roc_auc_score(y == 1, clf.predict_proba(x, n=n)[:, 1])


class Boosting:
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = (
            {} if base_model_params is None else base_model_params
        )

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []
        self.train_logits: list = []

        self.learning_rate: float = learning_rate
        self.early_stopping_rounds = early_stopping_rounds

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y / (1 + np.exp(y * z))

    def partial_fit(self, X, y):
        # y_hat = self.predict_logit(X)
        y_hat = (
            self.train_logits[-1]
            if len(self.train_logits)
            else np.zeros(y.shape[0], dtype=float)
        )
        s = -self.loss_derivative(y, y_hat)
        model = self.base_model_class(**self.base_model_params)
        model.fit(X, s)
        s_hat = model.predict(X)
        gamma = self.find_optimal_gamma(y, y_hat, s_hat)
        self.models.append(model)
        self.gammas.append(gamma)
        self.train_logits.append(y_hat + self.learning_rate * gamma * s_hat)

    def reset(self):
        self.models: list = []
        self.gammas: list = []

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        plot=False,
    ):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool
        """
        self.reset()
        for _ in range(self.n_estimators):
            self.partial_fit(X_train, y_train)
            self.history["train_roc_auc"].append(
                roc_auc_score(y_train == 1, self.sigmoid(self.train_logits[-1]))
            )
            self.history["train_loss"].append(
                self.loss_fn(y_train, self.train_logits[-1])
            )
            if X_val is not None and y_val is not None:
                val_logit = self.predict_logit(X_val)
                self.history["val_roc_auc"].append(
                    roc_auc_score(y_val == 1, self.sigmoid(val_logit))
                )
                self.history["val_loss"].append(self.loss_fn(y_val, val_logit))
                if self.early_stopping_rounds and self.early_stopping_rounds < len(
                    vh := self.history["val_loss"]
                ):
                    if all([hp <= hn for hp, hn in zip(vh[:-1], vh[1:])]):
                        break
        if plot:
            self.plot_history(X_val, y_val)

    def predict_logit(self, X, n=None):
        if 0 == len(self.models):
            return np.zeros(X.shape[0], dtype=float)
        n = len(self.models) if n is None else min(n, len(self.models))
        return self.learning_rate * np.hstack(
            [
                m.predict(X).reshape(-1, 1) * g
                for m, g in zip(self.models[:n], self.gammas[:n])
            ]
        ).sum(axis=1)

    def predict_proba(self, X, n=None):
        p = self.sigmoid(self.predict_logit(X, n=n)).reshape(-1, 1)
        return np.hstack([1.0 - p, p])

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0.01, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X, y, n=None):
        return score(self, X, y, n=n)

    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        if 0 == len(self.models):
            return
        logits = (
            self.learning_rate
            * np.hstack(
                [
                    m.predict(X).reshape(-1, 1) * g
                    for m, g in zip(self.models, self.gammas)
                ]
            )
            .cumsum(axis=1)
            .T
        )
        roc_auc_list = [roc_auc_score(y == 1, self.sigmoid(l)) for l in logits]
        loss_list = [self.loss_fn(y, z) for z in logits]

        fig_subplots = make_subplots(
            rows=1, cols=2, subplot_titles=["roc_auc", "loss"], vertical_spacing=0.05
        )
        fig1 = px.line(
            pl.DataFrame({"roc_auc": roc_auc_list}).with_row_index("iter", 1),
            "iter",
            "roc_auc",
        )
        fig2 = px.line(
            pl.DataFrame({"loss": loss_list}).with_row_index("iter", 1),
            "iter",
            "loss",
        )

        for trace in fig1.data:
            fig_subplots.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            fig_subplots.add_trace(trace, row=1, col=2)

        fig_subplots.update_layout(**layout_dict)
        fig_subplots.update_layout(width=1000)
        return fig_subplots
