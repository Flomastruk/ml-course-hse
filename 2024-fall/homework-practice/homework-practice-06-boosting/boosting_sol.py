from __future__ import annotations

from collections import defaultdict

import numpy as np
import plotly.express as px
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
        n_estimators: Optional[int] = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: Optional[int] = None,
        bootstrap_type: Optional[str] = None,
        subsample: float = 1.0,
        bagging_temperature: Optional[float] = 0.0,
        goss_k: Optional[float] = False,
        dart: bool = False,
        dropout_rate: Optional[float] = 0.05,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = (
            {} if base_model_params is None else base_model_params
        )
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        if bootstrap_type is not None:
            assert bootstrap_type in ["Bernoulli", "MVS"]
        self.bootstrap_type = bootstrap_type
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.goss_k = goss_k
        self.dart = dart
        self.dropout_rate = dropout_rate

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y / (1 + np.exp(y * z))

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}
        self.models: list = []
        self.gammas: list = []

        self.cached_predictions: dict = {}
        self.train_logits: Optional[np.ndarray] = None

    def get_subsample(self, X, y, y_hat):
        if self.bootstrap_type is None:
            return X, y, y_hat, None, -self.loss_derivative(y, y_hat), None

        l = X.shape[0]
        assert l == y.shape[0] and l == y_hat.shape[0]

        if self.goss_k is not None:
            s = -self.loss_derivative(y, y_hat)
            top_k = int(l * self.goss_k)
            other = int(l * self.subsample)

            inds = np.argpartition(np.abs(-s), top_k)
            inds = np.concatenate(
                [
                    inds[:top_k],
                    np.random.choice(
                        inds[top_k:],
                        size=other,
                        replace=False,
                    ),
                ]
            )
            w_ = np.concatenate(
                [
                    np.repeat(1.0, top_k),
                    np.repeat((1.0 - self.goss_k) / self.subsample, other),
                ]
            )
            return X[inds], y[inds], y_hat[inds], w_, s[inds], inds
        else:
            inds = np.random.choice(range(l), max(1, int(l * self.subsample), False))
            y_ = y[inds]
            y_hat_ = y_hat[inds]
            s_ = -self.loss_derivative(y_, y_hat_)
            return X[inds], y_, y_hat_, None, s_, inds

    def get_cached_prediction(self, i, X):
        if i not in self.cached_predictions:
            self.cached_predictions[i] = self.models[i].predict(X)
        return self.cached_predictions[i]

    def get_logits(self, X):
        if 0 == len(self.models):
            return (
                np.zeros(X.shape[0], dtype=float),
                [],
            )
        elif not self.dart:
            return self.train_logits, []
        else:
            dropout_models = [
                i
                for i, v in enumerate(
                    np.random.binomial(1, self.dropout_rate, size=len(self.models))
                )
                if v
            ]
            logits = (
                np.zeros(X.shape[0], dtype=float)
                if len(dropout_models) == len(self.models)
                else (
                    self.learning_rate
                    * np.hstack(
                        [
                            self.get_cached_prediction(i, X).reshape(-1, 1) * g
                            for i, g in enumerate(self.gammas)
                            if i not in dropout_models
                        ]
                    )
                    .sum(axis=1)
                    .T
                )
            )
            return logits, dropout_models

    def partial_fit(self, X, y):
        y_hat, dropout_models = self.get_logits(X)
        X_, y_, y_hat_, w_, s_, inds = self.get_subsample(X, y, y_hat)

        model = self.base_model_class(**self.base_model_params)
        if self.bagging_temperature == 0.0:
            model.fit(X_, s_, sample_weight=w_)
        else:
            w_ = -np.log(np.random.uniform(size=X_.shape[0]))
            if self.bagging_temperature != 1.0:
                w_ = np.power(w_, self.bagging_temperature)
            model.fit(X_, s_, sample_weight=w_)
        # full set
        s_hat = model.predict(X)
        s_hat_ = s_hat if inds is None else s_hat[inds]

        n_dropout_models = len(dropout_models)
        gamma = self.find_optimal_gamma(y_, y_hat_, s_hat_)
        if n_dropout_models > 0:
            gamma /= self.learning_rate + n_dropout_models
        self.gammas.append(gamma)
        self.models.append(model)
        scale = n_dropout_models / (self.learning_rate + n_dropout_models)
        for d in dropout_models:
            self.gammas[d] *= scale

        self.train_logits = (
            y_hat
            + (self.train_logits - y_hat) * scale
            + self.learning_rate * gamma * s_hat
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        n_epochs=None,
        plot=False,
        reset_cache=False,
    ):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool
        """
        if n_epochs is None:
            n_epochs = self.n_estimators

        if reset_cache:  # if X_train is a new datset, then need to reset
            self.cached_predictions = {}
            self.train_logits = (
                np.zeros(X_train.shape[0], dtype=float)
                if 0 == len(self.models)
                else (
                    self.learning_rate
                    * np.hstack(
                        [
                            self.get_cached_prediction(i, X_train).reshape(-1, 1) * g
                            for i, g in enumerate(self.gammas)
                        ]
                    )
                    .sum(axis=1)
                    .T
                )
            )
        if self.train_logits is None:
            self.train_logits = np.zeros(X_train.shape[0], dtype=float)
        for _ in range(n_epochs):
            self.partial_fit(X_train, y_train)
            self.history["train_roc_auc"].append(
                roc_auc_score(y_train == 1, self.sigmoid(self.train_logits))
            )
            self.history["train_loss"].append(self.loss_fn(y_train, self.train_logits))
            if X_val is not None and y_val is not None:
                val_logit = self.predict_logit(X_val)
                self.history["val_roc_auc"].append(
                    roc_auc_score(y_val == 1, self.sigmoid(val_logit))
                )
                self.history["val_loss"].append(self.loss_fn(y_val, val_logit))
                if self.early_stopping_rounds and self.early_stopping_rounds < len(
                    vh := self.history["val_loss"]
                ):
                    if all(
                        [
                            hp <= hn
                            for hp, hn in zip(
                                vh[-(self.early_stopping_rounds + 1) : -1],
                                vh[-self.early_stopping_rounds :],
                            )
                        ]
                    ):
                        break
        self.n_estimators = len(self.models)
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
