import datetime

import nltk
import numpy as np
import optuna
import polars as pl
import sklearn
import spacy

# from concurrent.futures import ProcessPoolExecutor
from fast_langdetect import detect
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
    cross_val_score,
    TimeSeriesSplit,
)
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
    # TargetEncoder,
)
from sklearn.svm import LinearSVC
from sklearn.utils import gen_batches, resample


sklearn.set_config(transform_output="polars")

try:
    nltk.data.find("misc/perluniprops")
except LookupError:
    print("Downloading perluniprops...")
    nltk.download("perluniprops")


def gini(y_true, y_score):
    return 2 * roc_auc_score(y_true, y_score) - 1.0


class GroupStatsImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_cols, target_cols, prefix="imputed_"):
        self.group_cols = group_cols
        self.target_cols = target_cols
        self.means_ = pl.DataFrame()
        self.global_mean_ = pl.DataFrame()
        self.prefix = prefix

    def fit(self, X, y=None):
        self.means_ = X.group_by(self.group_cols).agg(
            pl.col(self.target_cols).fill_nan(None).mean().name.prefix(self.prefix)
        )
        self.global_mean_ = X.select(
            pl.col(self.target_cols).fill_nan(None).mean().name.prefix(self.prefix)
        )
        return self

    def transform(self, X):
        return (
            X.join(self.global_mean_, how="cross")
            .update(self.means_, on=self.group_cols, how="left")
            .select(
                pl.col(self.target_cols)
                .fill_nan(None)
                .fill_null(pl.col([self.prefix + x for x in self.target_cols]))
                .name.prefix(self.prefix)
            )
        )


class HeroesEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        ohe = OneHotEncoder(
            categories=[list(range(1, 146))],
            handle_unknown="ignore",
            sparse_output=True,
        )
        ohe.set_output(transform="default")

        cols1 = [f"hero_{i}" for i in range(5)]
        cols2 = [f"hero_{i}" for i in range(128, 133)]
        return sum(ohe.fit_transform(X.select(c)) for c in cols1) - sum(
            ohe.fit_transform(X.select(c)) for c in cols2
        )


class NLTKtoSpacyTokenizer:
    def __init__(
        self, tokenizer: nltk.tokenize.api.TokenizerI, vocab: spacy.vocab.Vocab
    ):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __call__(self, text: str) -> spacy.tokens.doc.Doc:
        words = self.tokenizer.tokenize(text)
        if not words:
            return spacy.tokens.Doc(self.vocab, [], [])
        try:
            spaces = list(self.tokenizer.span_tokenize(text))
            spaces = [c[1] != n[0] for c, n in zip(spaces[:-1], spaces[1:])]
        except NotImplementedError:
            spaces = [True] * (len(words) - 1)

        spaces.append(text[-1] == " ")
        return spacy.tokens.doc.Doc(self.vocab, words, spaces)


def _process_match_df(
    df: pl.LazyFrame, column_transformer: ColumnTransformer | None = None
) -> tuple[pl.LazyFrame, ColumnTransformer]:
    # can't use month because there's only one year of data
    df = df.with_columns(
        month=pl.col("date").dt.month(),
        weekday=pl.col("date").dt.weekday(),
    ).with_columns(is_weekend=pl.col("weekday").ge(6))

    patch_dates = pl.LazyFrame(
        {
            "date": [
                datetime.date(2023, 12, 21),
                datetime.date(2024, 2, 21),
                datetime.date(2024, 3, 21),
                datetime.date(2024, 5, 22),
                datetime.date(2024, 7, 31),
                datetime.date(2024, 8, 28),
            ],
            "patch": ["735b", "735c", "735d", "736", "737", "737c"],
        }
    )
    df = df.join_asof(patch_dates, on="date")

    df = df.with_columns(
        avg_mmr_missing=pl.col("avg_mmr").is_null(),
        avg_mmr_q=pl.col("avg_mmr").fill_null(0.0),  # this will be modified later
        # avg_mmr_q=pl.col("avg_mmr").qcut(5, labels=[str(i) for i in range(5)])
    ).with_columns(
        avg_mmr_log1p=pl.col("avg_mmr").add(1.0).log(),
        avg_mmr_sqrt=pl.col("avg_mmr").pow(0.5),
        avg_mmr_recip=pl.col("avg_mmr").add(1).pow(-1.0),
    )

    passthrough = [
        "match_id",
        "date",
        "region",
        # "radiant_win",
        "game_mode",
        "duration",
        "month",
        "weekday",
        "is_weekend",
        "avg_mmr_missing",
    ]
    scaled = ["avg_mmr", "avg_mmr_log1p", "avg_mmr_sqrt", "avg_mmr_recip"]
    categorical = ["region", "patch"]
    if column_transformer is None:
        column_transformer = ColumnTransformer(
            [
                ("cols", "passthrough", passthrough),
                (
                    "scaler",
                    Pipeline(
                        [
                            (
                                "scaled_cols",
                                ColumnTransformer(
                                    [
                                        ("cols", "passthrough", ["region"]),
                                        (
                                            "scaler",
                                            StandardScaler(),
                                            scaled,
                                        ),
                                    ],
                                    verbose_feature_names_out=False,
                                ),
                            ),
                            ("scaled_imputed", GroupStatsImputer("region", scaled)),
                        ]
                    ),
                    ["region"] + scaled,
                ),
                (
                    "quantiler",
                    KBinsDiscretizer(
                        n_bins=5,
                        encode="ordinal",
                        strategy="quantile",
                        quantile_method="averaged_inverted_cdf",
                    ),
                    ["avg_mmr_q"],
                ),
                (
                    "ohe",
                    OneHotEncoder(
                        handle_unknown="ignore", drop="if_binary", sparse_output=False
                    ),
                    categorical,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        column_transformer.fit(df.collect())
    else:
        sklearn.utils.validation.check_is_fitted(column_transformer)

    df = column_transformer.transform(df.collect()).lazy()
    df = df.with_columns(
        avg_mmr_q=pl.when(pl.col("avg_mmr_missing"))
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(pl.col("avg_mmr_q"))
    )

    return df, column_transformer


def _fill_match_durations(df):
    return df.with_columns(duration=pl.lit(2546.0))  # simple train average for now


def process_match_df(
    is_train=True, column_transformer: ColumnTransformer | None = None
):
    df = pl.scan_csv(
        f"/data/ml-course-hse/ml1-2026-spring/homework-practice-03-features/matches_df_{'train' if is_train else 'test'}.csv",
        try_parse_dates=True,
    ).sort("date")
    if not is_train:
        df = _fill_match_durations(df)
        df = df.with_columns(radiant_win=pl.lit(None).cast(pl.Boolean))

    df = _process_match_df(df, column_transformer=column_transformer)
    return df


def process_player_df() -> pl.LazyFrame:
    players = pl.scan_csv(
        "/data/ml-course-hse/ml1-2026-spring/homework-practice-03-features/player_df.csv",
        try_parse_dates=True,
    )
    # slot conflicts -- this is bad data
    players = players.unique(subset=["player_slot", "match_id"], keep="last")

    bad_match_id = pl.concat(
        [
            (
                # 1000 matches with incorrect hero counts
                players.group_by("match_id", "hero_id")
                .agg(hero_count=pl.len())
                .group_by("match_id")
                .agg(pl.col("hero_count").max())
                .filter(pl.col("hero_count") > 1)
                .select("match_id")
            ),
            (
                # matches with hero_id==0
                players.filter(pl.col("hero_id") == 0).select(
                    pl.col("match_id").unique()
                )
            ),
            (
                # matches with same account appearing twice
                players.filter(~pl.col("account_id").is_in([-1, 4294967295]))
                .group_by("match_id", "account_id")
                .agg(acct_count=pl.len())
                .group_by("match_id")
                .agg(pl.col("acct_count").max())
                .filter(pl.col("acct_count") > 1)
                .select("match_id")
            ),
        ]
    )
    # return players
    return players.join(bad_match_id, on="match_id", how="anti")


def process_hero_df() -> pl.LazyFrame:
    heroes = pl.scan_csv(
        "/data/ml-course-hse/ml1-2026-spring/homework-practice-03-features/Constants.Heroes.csv",
        try_parse_dates=True,
    )
    return heroes


def nlp_preprocess_factory(tokenizer: nltk.tokenize.api.TokenizerI = None):
    def preprocess_batch(texts: list[str], model: str) -> list[str]:
        nlp = spacy.load(
            model,
            enable=["attribute_ruler", "lemmatizer", "tagger", "tok2vec"],
        )
        if tokenizer:
            nlp.tokenizer = NLTKtoSpacyTokenizer(tokenizer, nlp.vocab)
        if "lemmatizer" in nlp.pipe_names:
            return [[token.lemma_ for token in doc] for doc in nlp.pipe(texts)]
        else:
            return [[token.text for token in doc] for doc in nlp.pipe(texts)]

    return preprocess_batch


def _process_chat_df(
    chats: pl.LazyFrame,
    chat_name: pl.LazyFrame,
    tokenizer: nltk.tokenize.api.TokenizerI = None,
) -> pl.LazyFrame:
    assert chat_name in ("radiant_chat", "dire_chat"), "Unsupported chat"

    web_model_langs = ["en", "zh"]
    news_model_langs = [
        "ca",
        "da",
        "de",
        "el",
        "es",
        "fi",
        "fr",
        "hr",
        "it",
        "ja",
        "ko",
        "lt",
        "mk",
        "nb",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sl",
        "sv",
        "uk",
    ]

    msgs = (
        chats.filter(~pl.col(chat_name).is_null())
        .select(
            "match_id",
            msg=pl.col(chat_name)
            .str.to_lowercase()
            .str.strip_chars(" ")
            .str.replace_all(r"\?{2,}", "?")
            .str.replace_all(r"!{2,}", "!")
            .str.replace_all(r"\.{2,}", ".")
            .str.replace_all(r"\({2,}", "(")
            .str.replace_all(r"\){2,}", ")")
            .str.split("|"),
        )
        .explode("msg")
        .with_columns(
            lang=pl.col("msg").map_elements(
                lambda x: detect(x)[0]["lang"], return_dtype=pl.String
            )
        )
        .with_columns(
            model=pl.when(pl.col("lang").is_in(web_model_langs))
            .then(pl.col("lang") + "_core_web_sm")
            .when(pl.col("lang").is_in(news_model_langs))
            .then(pl.col("lang") + "_core_news_sm")
            .otherwise(pl.lit("xx_sent_ud_sm"))
        )
    )
    # .collect()  # need to collect here otherwise too slow

    for model in msgs.select(pl.col("model").unique()).collect()["model"]:
        if not spacy.util.is_package(model):
            spacy.cli.download(model)

    preprocess_batch = nlp_preprocess_factory(tokenizer=tokenizer)
    msgs = (
        (
            msgs.group_by("model")
            .agg(pl.all())
            .with_columns(
                normalized_msg=pl.struct("model", "msg").map_elements(
                    lambda x: preprocess_batch(x["msg"], x["model"]),
                    return_dtype=pl.List(pl.List(pl.String)),
                )
            )
        )
        .explode("match_id", "msg", "normalized_msg", "lang")
        .group_by("match_id")
        .agg(pl.all())
        .with_columns(pl.col("normalized_msg").list.eval(pl.element().explode()))
        .collect()
    )
    return msgs


def process_chat_df(tokenizer: nltk.tokenize.api.TokenizerI = None) -> pl.LazyFrame:
    chats = pl.scan_csv(
        f"/data/ml-course-hse/ml1-2026-spring/homework-practice-03-features/game_chat.csv",
        try_parse_dates=True,
    )
    radiant_chats = _process_chat_df(chats, "radiant_chat", tokenizer=tokenizer).select(
        pl.col("match_id"), pl.col("normalized_msg").alias("radiant_chat_norm")
    )
    dire_chats = _process_chat_df(chats, "dire_chat", tokenizer=tokenizer).select(
        pl.col("match_id"), pl.col("normalized_msg").alias("dire_chat_norm")
    )

    return radiant_chats.join(dire_chats, on="match_id", how="full", coalesce=True)


def combine_dfs(df: pl.LazyFrame, players: pl.LazyFrame) -> pl.LazyFrame:
    # anti-pattern: we'll eliminate df records without matches
    # at the time of prediction we'll fill bad data with dummies. Not worth the hassle
    slot_heroes = players.pivot(
        on="player_slot",
        on_columns=[0, 1, 2, 3, 4, 128, 129, 130, 131, 132],
        index="match_id",
        values="hero_id",
    ).select("match_id", pl.selectors.matches(r"^\d").name.prefix("hero_"))
    df = df.join(slot_heroes, how="left", on="match_id")
    henc = HeroesEncoder()
    # TODO: this is fast operation that collects a complex dataframe, can be a part of a pipeline that works on prestored data
    df = pl.concat(
        [
            df.drop(pl.selectors.starts_with("hero_")),
            pl.DataFrame(
                henc.transform(
                    df.select(pl.selectors.starts_with("hero_")).collect()
                ).toarray(),
                schema=[f"hero_{i}" for i in range(1, 146)],
            ).lazy(),
        ],
        how="horizontal",
    )

    return df


def save_parquet():
    pass


def gen_objective(X: pl.DataFrame, y: pl.Series):
    # cols = (
    #     pl.selectors.starts_with("region_"),
    #     pl.selectors.starts_with("hero_"),
    #     "is_weekend",
    #     "mmr_missing",
    #     "avg_mmr_log1p",
    # )
    # X = df.select(*cols).collect()
    # y = df.select("radiant_win").collect()["radiant_win"]
    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical("model_type", ["logr", "svc"])

        match model_type:
            case "logr":
                C = trial.suggest_float("C", 1e-3, 1e6, log=True)
                max_iter = trial.suggest_int("max_iter", 100, 1000, log=True)
                model = LogisticRegression(C=C, max_iter=max_iter)
            case "svc":
                C = trial.suggest_float("C", 1e-3, 1e6, log=True)
                max_iter = trial.suggest_int("max_iter", 100, 1000, log=True)
                loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
                if loss == "hinge":
                    max_iter *= 2
                model = LinearSVC(C=C, max_iter=max_iter, loss=loss)
            case _:
                raise NotImplementedError

        cv_time = TimeSeriesSplit(n_splits=4)
        y_hat = np.concatenate(
            [
                model.fit(X[tr], y[tr]).decision_function(X[ts])
                for tr, ts in cv_time.split(X)
            ]
        )
        # y_hat = cross_val_predict(model, X, y, cv=cv_time, method="decision_function") # doesn't work because TimeSeriesSplit isn't a disjoint partition

        # res = cross_validate(
        #     model, X, y,
        #     cv=cv_time,
        #     scoring="roc_auc",
        #     return_estimato=True,
        # )
        # p_hat = res['estimator'][-1].decision_function(X)
        # trial.set_user_attr("accuracy", accuracy_score(y_val, 0.5 <= p_hat))

        return gini(y[-len(y_hat) :], y_hat)

    return objective


# optuna trials with SGDClassifier and stopping criteria
def gen_objective_sgd(X: pl.DataFrame, y: pl.Series):
    def objective_sgd(trial: optuna.Trial) -> float:
        n_splits = 4
        cv_time = TimeSeriesSplit(n_splits=n_splits)

        loss = trial.suggest_categorical("loss", ["log_loss", "squared_hinge", "hinge"])
        learning_rate = trial.suggest_categorical(
            "learning_rate", ["optimal", "invscaling"]
        )
        models = [
            SGDClassifier(loss=loss, learning_rate=learning_rate, shuffle=False)
            for _ in range(n_splits)
        ]

        max_iter = 8
        n_batches = 1  # make as few batches as possible -- as long as it fits in memory, this speeds up the process
        step = 0
        for _ in range(max_iter):
            train_inds = [
                resample(
                    tr,
                    replace=False,
                    n_samples=len(tr),
                    random_state=trial.number + 123,
                )
                for tr, _ in cv_time.split(X)
            ]
            for batches in zip(
                *(gen_batches(len(tr), -(-len(tr) // n_batches)) for tr in train_inds)
            ):
                for model, (X_, y_) in zip(
                    models,
                    ((X[ind := tr[b]], y[ind]) for tr, b in zip(train_inds, batches)),
                ):
                    model.partial_fit(X_, y_, classes=[False, True])

                # # this is actually much slower becase it pickles stuff.. so need to have a proper pipeline or something better.
                # Xy = zip(*((X[ind := tr[b]], y[ind]) for tr, b in zip(train_inds, batches)))
                # with ProcessPoolExecutor(max_workers=n_splits) as executor:
                #     models = list(executor.map(job_func_full, zip(models, *Xy)))

                g = (
                    sum(
                        gini(y[ts], model.decision_function(X[ts]))
                        for model, (_, ts) in zip(models, cv_time.split(X))
                    )
                    / n_splits
                )
                print(f"beginning step: {step}, gini: {g}")
                step += 1
                trial.report(g, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        return g

    return objective_sgd


class unpack:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        args = list(args)
        if args and isinstance(args[-1], (list, tuple)):
            args.extend(args.pop())
        return self.func(*args, **kwargs)


def job_func_full_(model: SGDClassifier, X: pl.DataFrame, y: pl.Series) -> dict:
    """Utility for parallel processing"""
    try:
        model.partial_fit(
            X,
            y,
            classes=[False, True],
        )
        return model
    except RuntimeError:
        return {}


job_func_full = unpack(job_func_full_)
