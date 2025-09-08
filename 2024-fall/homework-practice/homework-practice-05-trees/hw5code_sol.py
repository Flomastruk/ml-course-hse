import numpy as np
import polars as pl

from sklearn.base import BaseEstimator


def find_best_split(feature_vector: pl.Series, target_vector: pl.Series):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    assert len(feature_vector) == len(target_vector), "Inputs must have equal lengths"
    if 0 == len(feature_vector):
        return pl.Series([], dtype=float), pl.Series([], dtype=float), None, None

    sel_left = pl.selectors.starts_with("left_")
    sel_right = pl.selectors.starts_with("right_")

    res = (
        pl.DataFrame({"f": feature_vector, "v": target_vector})
        .group_by(pl.col("f", "v"))
        .agg(pl.len().alias("count"))
        .sort("f")
        .pivot(
            on="v",
            index="f",
            values="count",
        )
        .with_columns(
            pl.exclude("f").fill_null(0),
        )
        .select(
            (0.5 * (pl.col("f") + pl.col("f").shift(-1))).alias("threshold"),
            pl.exclude("f").cum_sum().name.prefix("left_"),
            (pl.exclude("f").sum() - pl.exclude("f").cum_sum()).name.prefix("right_"),
        )[:-1]
        .with_columns(
            pl.sum_horizontal(sel_left).alias("total_left"),
            pl.sum_horizontal(sel_right).alias("total_right"),
        )
        .with_columns(
            h_l=1.0 - pl.sum_horizontal(sel_left.truediv("total_left").pow(2)),
            h_r=1.0 - pl.sum_horizontal(sel_right.truediv("total_right").pow(2)),
        )
        .select(
            pl.col("threshold"),
            q=(pl.col("h_l").mul("total_left") + pl.col("h_r").mul("total_right"))
            .truediv(pl.col("total_left") + pl.col("total_right"))
            .neg(),
        )
    )
    # .with_columns( # doesn't work as expected
    #     (sel_right - sel_left)
    # )
    best_ix = res["q"].arg_max()

    return res["threshold"], res["q"], res[best_ix, "threshold"], res[best_ix, "q"]


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if np.any(
            list(
                map(
                    lambda x: (x != "real") and (x != "categorical"),
                    feature_types.values(),
                )
            )
        ):
            raise ValueError("There is unknown feature type")

        self._feature_types = feature_types
        self.tree_ = {"depth": 0}
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split if min_samples_split else 2
        self._min_samples_leaf = min_samples_leaf if min_samples_leaf else 1

    def _fit_node(self, sub_X, sub_y, node):
        is_terminal = False
        is_terminal |= node["depth"] >= self._max_depth
        is_terminal |= (sub_y == sub_y[0]).all()
        is_terminal |= self._min_samples_split > len(sub_y)

        if is_terminal:
            node["type"] = "terminal"
            node["class"] = (
                sub_y.median() if sub_y.dtype.is_numeric() else sub_y.mode().first()
            )
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in sub_X.columns:
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[feature]
            elif feature_type == "categorical":
                categories_map = (
                    sub_X.with_columns(y=sub_y)
                    .group_by(feature)
                    .agg(pl.col("y").mean().alias("avg"))
                    .select(feature, rk=pl.col("avg").rank(method="ordinal"))
                )
                feature_vector = sub_X.join(categories_map, on=feature)["rk"]
            else:
                raise ValueError

            if feature_vector.n_unique() == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                split = feature_vector < threshold
                if self._min_samples_split and sum(split) < self._min_samples_leaf:
                    # EA technically need to run through all thresholds but we will skip feature in this setting
                    continue
                feature_best = feature
                gini_best = gini

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "Categorical":
                    threshold_best = list(
                        categories_map.filter(pl.col("rk") < threshold)[feature_best]
                    )
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = (
                sub_y.median() if sub_y.dtype.is_numeric() else sub_y.mode().first()
            )
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {"depth": node["depth"] + 1}, {
            "depth": node["depth"] + 1
        }
        self._fit_node(sub_X.filter(split), sub_y.filter(split), node["left_child"])
        self._fit_node(
            sub_X.filter(split.not_()),
            sub_y.filter(split.not_()),
            node["right_child"],
        )

    def _predict_node(self, x, node):
        if node is None:
            return 99  # dummy
        return x.select(
            pred=pl.when(node["type"] == "terminal")
            .then(pl.lit(node["class"]) if node["type"] == "terminal" else 99)
            .when(
                False
                if node["type"] == "terminal"
                else (
                    pl.col(node["feature_split"]) <= node["threshold"]
                    if self._feature_types[node["feature_split"]] == "real"
                    else pl.col(node["feature_split"]).is_in(node["categories_split"])
                )
            )
            .then(
                self._predict_node(x, node["left_child"])
                if node["type"] == "nonterminal"
                else pl.repeat(99, pl.len())
            )
            .otherwise(
                self._predict_node(x, node["right_child"])
                if node["type"] == "nonterminal"
                else pl.repeat(99, pl.len())
            )
        )["pred"]

    def fit(self, X, y):
        self._fit_node(X, y, self.tree_)
        return self

    def predict(self, X):
        return self._predict_node(X, self.tree_)


def find_best_split_linreg(
    feature_splits: pl.Series,
    feature_vector: pl.Series,
    target_vector: pl.Series,
):
    """
    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return losses: вектор со значениями для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return losses_best: оптимальное значение (число)
    """
    assert len(feature_vector) == len(target_vector), "Inputs must have equal lengths"
    assert len(feature_splits) == len(target_vector), "Inputs must have equal lengths"
    if 0 == len(feature_splits):
        return pl.Series([], dtype=float), pl.Series([], dtype=float), None, None
    assert feature_splits.n_unique() > 1, "This function mustn't be called this way"

    splits = (
        pl.DataFrame({"s": feature_splits, "f": feature_vector, "v": target_vector})
        .group_by(pl.col("s"))
        .agg(fs=pl.col("f"), vs=pl.col("v"))
        .sort("s")
        .with_row_index("ix", 1)
        .select(pl.col("ix", "s"), pl.col("fs").implode(), pl.col("vs").implode())[:-1]
        .select(
            pl.col("ix", "s"),
            left_fs=pl.col("fs").list.head(pl.col("ix")),
            left_vs=pl.col("vs").list.head(pl.col("ix")),
            right_fs=pl.col("fs").list.slice(pl.col("ix"), None),
            right_vs=pl.col("vs").list.slice(pl.col("ix"), None),
        )
        .with_columns(
            pl.col("left_fs", "left_vs", "right_fs", "right_vs").list.eval(
                pl.element().explode()
            )
        )
    )
    left_splits = (
        (
            splits.select(pl.exclude("right_fs", "right_vs"))
            .explode(columns=["left_fs", "left_vs"])
            .with_columns(
                xy=pl.cov("left_fs", "left_vs").over("ix").fill_null(0.0),
                xx=pl.var("left_fs").over("ix").fill_null(0.0),
            )
            .with_columns(
                b=pl.when(pl.col("xx") == 0.0)
                .then(0.0)
                .otherwise(pl.col("xy").truediv("xx"))
            )
            .with_columns(
                a=pl.col("left_vs").sub(pl.col("b").mul("left_fs")).mean().over("ix")
            )
        )
        .group_by("ix")
        .agg(
            max_f_left=pl.col("left_fs").max(),
            n_left=pl.len(),
            loss_left=pl.col("left_vs")
            .sub(pl.col("a") + pl.col("b").mul("left_fs"))
            .pow(2)
            .mean(),
        )
    )

    right_splits = (
        (
            splits.select(pl.exclude("left_fs", "left_vs"))
            .explode(columns=["right_fs", "right_vs"])
            .with_columns(
                xy=pl.cov("right_fs", "right_vs").over("ix").fill_null(0.0),
                xx=pl.var("right_fs").over("ix").fill_null(0.0),
            )
            .with_columns(
                b=pl.when(pl.col("xx") == 0.0)
                .then(0.0)
                .otherwise(pl.col("xy").truediv("xx"))
            )
            .with_columns(
                a=pl.col("right_vs").sub(pl.col("b").mul("right_fs")).mean().over("ix")
            )
        )
        .group_by("ix")
        .agg(
            min_f_right=pl.col("right_fs").min(),
            n_right=pl.len(),
            loss_right=pl.col("right_vs")
            .sub(pl.col("a") + pl.col("b").mul("right_fs"))
            .pow(2)
            .mean(),
        )
    )
    res = left_splits.join(right_splits, on="ix").with_columns(
        threshold=pl.col("max_f_left").add(pl.col("min_f_right")).mul(0.5),
        loss=(
            pl.col("n_left").mul("loss_left") + pl.col("n_right").mul("loss_right")
        ).truediv(pl.col("n_left").add(pl.col("n_right"))),
    )
    best_ix = res["loss"].arg_min()

    return (
        res["threshold"],
        res["loss"],
        res[best_ix, "threshold"],
        res[best_ix, "loss"],
    )


class LinearRegressionTree(BaseEstimator):
    def __init__(
        self,
        # base_model_type=None, # EA: will only implement MSE
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_split_quantiles=None,
    ):
        self.tree_ = {"depth": 0}
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_split_quantiles = n_split_quantiles

    def _fit_node(self, sub_X, sub_y, node: dict):
        is_terminal = False
        is_terminal |= node["depth"] >= self.max_depth
        is_terminal |= sub_y.n_unique() == 1
        is_terminal |= self.min_samples_split > len(sub_y)

        feature_best, threshold_best, loss_best, split = (
            None,
            None,
            None,
            None,
        )
        for feature in [] if is_terminal else sub_X.columns:
            feature_vector = sub_X[feature]
            if self.n_split_quantiles is not None:
                feature_splits = (
                    (feature_vector.rank("dense") - 1)
                    / feature_vector.n_unique()
                    * self.n_split_quantiles
                ).floor()
            else:
                feature_splits = feature_vector

            if feature_splits.n_unique() <= 1:
                continue
            _, _, threshold, _loss = find_best_split_linreg(
                feature_splits, feature_vector, sub_y
            )
            if loss_best is None or _loss < loss_best:
                split = feature_vector <= threshold
                if self.min_samples_split and sum(split) < self.min_samples_leaf:
                    # EA technically need to run through all thresholds but we will skip feature in this setting
                    continue
                loss_best = _loss
                feature_best = feature
                threshold_best = threshold
                assert (
                    threshold_best is not None
                ), "This never happens since feature_splits.n_unique >= 1"

        if feature_best is None:
            loss_best, a_best, b_best = None, None, None
            for feature in sub_X.columns:
                feature_vector = sub_X[feature]
                v = feature_vector.var()
                if v == 0.0 or v is None:
                    a = sub_y.mean()
                    b = 0.0
                else:
                    b = pl.cov(feature_vector, sub_y, eager=True).item() / v
                    a = (sub_y - b * feature_vector).mean()
                _loss = (sub_y - a - b * feature_vector).pow(2).mean()
                if loss_best is None or _loss < loss_best:
                    loss_best = _loss
                    feature_best, a_best, b_best = feature, a, b
            node["type"] = "terminal"
            node["covariate_name"] = feature_best
            node["a"] = a_best
            node["b"] = b_best
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {"depth": node["depth"] + 1}, {
            "depth": node["depth"] + 1
        }
        self._fit_node(sub_X.filter(split), sub_y.filter(split), node["left_child"])
        self._fit_node(
            sub_X.filter(split.not_()),
            sub_y.filter(split.not_()),
            node["right_child"],
        )

    def _predict_node(self, x, node):
        if node is None:
            return 99  # dummy
        return x.select(
            pred=pl.when(node["type"] == "terminal")
            .then(
                pl.col(node["covariate_name"]).mul(node["b"]).add(node["a"])
                if node["type"] == "terminal"
                else 99
            )
            .when(
                False
                if node["type"] == "terminal"
                else pl.col(node["feature_split"]) <= node["threshold"]
            )
            .then(
                self._predict_node(x, node["left_child"])
                if node["type"] == "nonterminal"
                else pl.repeat(99, pl.len())
            )
            .otherwise(
                self._predict_node(x, node["right_child"])
                if node["type"] == "nonterminal"
                else pl.repeat(99, pl.len())
            )
        )["pred"]

    def fit(self, X, y):
        self._fit_node(X, y, self.tree_)
        return self

    def predict(self, X):
        return self._predict_node(X, self.tree_)
