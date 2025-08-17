from collections import Counter

import numpy as np
import polars as pl


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

        self.tree_ = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split if min_samples_split else 1
        self._min_samples_leaf = min_samples_leaf if min_samples_leaf else 1

    def _fit_node(self, sub_X, sub_y, node):
        is_terminal = False
        is_terminal |= (sub_y == sub_y[0]).all()
        is_terminal |= self._min_samples_split >= len(sub_y)

        if is_terminal:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
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
        node["left_child"], node["right_child"] = {}, {}
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
                    pl.col(node["feature_split"]) < node["threshold"]
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

    def predict(self, X):
        return self._predict_node(X, self.tree_)


class LinearRegressionTree:
    def __init__(
        self,
        feature_types,
        base_model_type=None,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        pass
