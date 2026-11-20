from concurrent.futures import ProcessPoolExecutor
from enum import Enum, auto
from functools import partial

import numpy as np
import polars as pl


def get_cohen_kappa(y, y_hat):
    po = (y == y_hat).mean()
    pe = y.mean() * y_hat.mean() + (1.0 - y).mean() * (1.0 - y_hat).mean()

    return (po - pe) / (1.0 - pe)


# Note: this is simply the Pearson correlation between y and y_hat
def get_matthews_corrcoef(y, y_hat):
    err_matrix = (
        pl.DataFrame({"y": y, "y_hat": y_hat})
        .pivot(
            on="y",
            index="y_hat",
            values="y_hat",
            aggregate_function=pl.len(),
            sort_columns=True,
        )
        .sort("y_hat")
    )
    if pl.Series(y).n_unique() < 2 or pl.Series(y_hat).n_unique() < 2:
        return 0.0

    # note that the order is ill-defined, but the output is correct regardless
    tp = err_matrix[1, 2]
    tn = err_matrix[0, 1]
    fp = err_matrix[1, 1]
    fn = err_matrix[0, 2]

    return (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5


class NodeType(Enum):
    INTERNAL = auto()
    EXTERNAL = auto()


def c_factor(n):
    """
    Computes average path length for an unsuccessful search in a binary search tree.
    Params:
        n: int - number of data points for BST
    """
    return (2.0 / np.arange(1, n)).sum() - 2 * (n - 1) / n


class Node:
    """
    A single node object for each tree. Contains information on height, current data,
    splitting hyperplane and children nodes.

    Attributes:
        X: np.array - data available to current node
        size: int - length of available data

        depth: int - depth of node

        left: Node - left child
        right: Node - right child

        kind: str - either "internal" or "external", indicates the type of current node

        w: np.array - normal vector for the splitting hyperplane
        b: float - intercept term for the splitting hyperplane
    """

    def __init__(
        self,
        X: np.ndarray,
        depth: int,
        left: "Node | None",
        right: "Node | None",
        w: np.ndarray | None,
        b: np.int64 | None,
        kind: NodeType,
    ):
        """
        Node(h, left, right, kind, w, b)
        Represents the node object.

        Params:
            X: np.array - data available to current node
            depth: int - depth of node

            left: Node - left child
            right: Node - right child

            kind: str - either "internal" or "external", indicates the type of current node

            w: np.array - normal vector for the splitting hyperplane
            b: float - intercept term for the splitting hyperplane

        """
        self.size = len(X)

        self.depth = depth

        self.left = left
        self.right = right

        self.kind = kind

        self.w = w
        self.b = b

    def __repr__(self):
        """
        For convenience only.
        """
        return f"Node(size={self.size}, depth={self.depth}, kind={self.kind})"


def calc_height(
    X: np.ndarray,
    node: Node,
    out: np.ndarray | None = None,
    idx: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculates anomaly scores for sample in a recursive manner.
    Params:
        X: np.array - current sample, available to node

        depth: int - path length up to current node

        node: Node - current tree node

    Returns:
        scores: int, float or np.array - anomaly scores for sample
    """
    if out is None:
        out = np.zeros(len(X))
        idx = np.arange(len(X))

    if node.kind == NodeType.EXTERNAL:
        out[idx] = node.depth
        return out

    left_ix = X[idx] @ node.w + node.b < 0
    calc_height(X, node.left, out, idx[left_ix])
    calc_height(X, node.right, out, idx[~left_ix])
    return out


class RandomizedTree:
    """
    Single randomized tree object. Stores root and its depth (tree is built recursively).
    Attributes:
        depth: int - current tree depth

        max_depth: int - maximum tree depth

        root: Node - root node

        internal_count: int - number of internal nodes

        external_count: int - number of external nodes

    """

    def __init__(self, X, max_depth, rng):
        """
        Single randomized tree object. Stores root and its depth (tree is built recursively).
        Params:
            X: np.array - train sample
            max_depth: int - maximum tree depth

        """
        self.depth = 0
        self.max_depth = max_depth

        self.fit_size = len(X)
        self.internal_count = 0
        self.external_count = 0

        self.root = self.grow(X, 0, rng)

    def __repr__(self):
        """
        For convenience only.
        """

        return f"RandomizedTree(depth={self.depth}, max_depth={self.max_depth}, n_internal={self.internal_count}, n_external={self.external_count})"

    def grow(
        self, X: np.ndarray, depth: int, rng: np.random.Generator | int | None
    ) -> Node | None:
        """
        Grow tree in a recursive manner.
        Params:
            X: np.array - available train sample

            depth: int - current tree depth

        Returns:
            node: Node - a trained node with separating hyperplane data.
                         Node provides access to children if necessary (these are built recursively)
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        if depth == self.max_depth or len(np.unique(X, axis=0)) <= 1:
            self.external_count += 1
            return Node(X, depth, None, None, None, None, NodeType.EXTERNAL)

        w = rng.multivariate_normal(mean=np.zeros(X.shape[-1]), cov=np.eye(X.shape[-1]))
        w /= np.linalg.norm(w)
        Xw = X @ w
        b = -rng.choice(Xw[Xw > Xw.min()])

        left_ix = Xw + b < 0
        left = self.grow(X[left_ix], depth + 1, rng)
        right = self.grow(X[~left_ix], depth + 1, rng)

        self.internal_count += 1
        return Node(X, depth, left, right, w, b, NodeType.INTERNAL)

    def score_samples(self, X):
        """
        Calculate anomaly scores for given data. You may utilize outer function `calc_height`.
        Params:
            X: np.array - data to be evaluated

        Returns:
            scores: np.array - estimated anomaly scores
        """
        numerator = calc_height(X, self.root)
        return 2 ** (-numerator / c_factor(self.fit_size))


def fit_tree(
    X: np.ndarray,
    subsample_rate: float,
    max_depth: int | None,
    rng: np.random.Generator,
) -> RandomizedTree:
    mask = rng.binomial(n=1, p=subsample_rate, size=len(X)).astype(bool)
    return RandomizedTree(X[mask], max_depth, rng=rng)


class ExtendedIsolationForest:
    """
    Extended Isolation Forest object. Stores training data and trained randomized trees.
    Attributes:
        n_trees: int - number of Randomized Trees

        max_depth: int - maximum depth of each tree

        subsample_rate: float - draw `subsample_rate * X.shape[0]` samples for each tree

        trees: list - container for trained trees

        contamination: float - estimated fraction of anomaly samples in data. Used for thresholding

    """

    def __init__(self, n_trees, subsample_rate, max_depth=None, contamination=0.01):
        """
        Extended Isolation Forest object. Stores training data and trained randomized trees.
        Params:
            n_trees: int - number of Randomized Trees

            subsample_rate: float - draw `subsample_rate * X.shape[0]` samples for each tree

            max_depth: int or None - maximum depth of each tree. Defaults to ceil(log_2(subsample_size)) if not provided

            contamination: float - estimated fraction of anomaly samples in data. Used for thresholding

        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.subsample_rate = subsample_rate
        self.trees: list[RandomizedTree] = []
        self.contamination: float = contamination
        self.threshold: float | None = None
        self.is_fit = False

    def __repr__(self):
        """For convenience only."""

        return f"ExtendedIsolationForest(n_trees={self.n_trees}, max_depth={self.max_depth}, subsample_rate={self.subsample_rate}, contamination={self.contamination}, is_fit={self.is_fit})"

    def fit(self, X, rng=None, n_jobs=None):
        """
        Fit EIF to new data.
        Params:
            X: np.array - 2d array of samples
        """
        rng = np.random.default_rng(rng)

        if n_jobs is not None and n_jobs > 1:
            fit_one = partial(fit_tree, X, self.subsample_rate, self.max_depth)
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                self.trees = list(
                    ex.map(
                        fit_one,
                        rng.spawn(self.n_trees),
                        chunksize=-(-self.n_trees // n_jobs),
                    )
                )
        else:
            self.trees = [
                fit_tree(X, self.subsample_rate, self.max_depth, r)
                for r in rng.spawn(self.n_trees)
            ]
            # for _ in range(self.n_trees):
            #     X_sub = X[rng.binomial(1, self.subsample_rate, len(X)).astype(bool)]
            #     tree = RandomizedTree(X_sub, self.max_depth, rng=rng)
            #     self.trees.append(tree)

        scores = self.score_samples(X, n_jobs=n_jobs)
        self.threshold = np.quantile(scores, q=1.0 - self.contamination)
        self.is_fit = True

        return self

    def score_samples(self, X, n_jobs=None):
        """
        Estimate (normalized) anomaly score for each given sample
        Params:
            X: np.array - new samples

        Returns:
            scores: np.array - anomaly scores (larger value means higher probability of a sample being an outlier)
        """
        if n_jobs is not None and n_jobs > 1:
            score_one = partial(calc_height, X)
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                numerator = np.stack(
                    list(
                        ex.map(
                            score_one,
                            [t.root for t in self.trees],
                            chunksize=-(-self.n_trees // n_jobs),
                        )
                    ),
                    axis=1,
                ).mean(axis=-1)
        else:
            numerator = np.stack(
                [calc_height(X, t.root) for t in self.trees], axis=1
            ).mean(axis=-1)
        denominator = np.array([t.fit_size for t in self.trees]).mean()
        return 2 ** (-numerator / c_factor(denominator))

    def predict(self, X, n_jobs=None):
        """
        Predict if given samples are outliers.
        Params:
            X: np.array - new samples

        Returns:
            labels: np.array - anomaly labels (1 for outliers, 0 for inliers)
        """

        return self.score_samples(X, n_jobs=n_jobs) >= self.threshold
