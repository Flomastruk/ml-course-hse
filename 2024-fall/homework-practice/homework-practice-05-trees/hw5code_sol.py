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
    if not len(feature_vector):
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
            list(map(lambda x: x != "real" and x != "categorical", feature_types))
        ):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y != sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(1, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / current_click
                sorted_categories = list(
                    map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1]))
                )
                categories_map = dict(
                    zip(sorted_categories, list(range(len(sorted_categories))))
                )

                feature_vector = np.array(
                    map(lambda x: categories_map[x], sub_X[:, feature])
                )
            else:
                raise ValueError

            if len(feature_vector) == 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "Categorical":
                    threshold_best = list(
                        map(
                            lambda x: x[0],
                            filter(lambda x: x[1] < threshold, categories_map.items()),
                        )
                    )
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)
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
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[split], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        pass

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


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
