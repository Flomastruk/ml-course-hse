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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


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


def get_tokenized_chats(tokenizer_name: str, force_recreate=False) -> pl.DataFrame:
    """Return tokenized chats, if"""
    assert tokenizer_name in ("toktok", "nist", "destructor", "tweet")

    import os

    target_path = f"/data/ml-course-hse/ml1-2026-spring/chats_{tokenizer_name}.parquet"
    if not force_recreate and os.path.exists(target_path):
        return pl.read_parquet(target_path)

    chats = pl.scan_csv(
        f"/data/ml-course-hse/ml1-2026-spring/homework-practice-03-features/game_chat.csv",
        try_parse_dates=True,
    )
    match tokenizer_name:
        case "toktok":
            tokenizer = nltk.tokenize.ToktokTokenizer()
        case "nist":
            from nltk.tokenize.nist import NISTTokenizer

            tokenizer = NISTTokenizer()
        case "destructor":
            tokenizer = nltk.tokenize.destructive.NLTKWordTokenizer()
        case "tweet":
            tokenizer = nltk.tokenize.casual.TweetTokenizer(
                reduce_len=True, preserve_case=False
            )

    msgs = (
        _process_chat_df(chats, "radiant_chat", tokenizer)
        .select(pl.col("match_id"), pl.col("normalized_msg").alias("radiant_chat_norm"))
        .join(
            _process_chat_df(chats, "dire_chat", tokenizer).select(
                pl.col("match_id"), pl.col("normalized_msg").alias("dire_chat_norm")
            ),
            on="match_id",
            how="full",
            coalesce=True,
        )
    )

    msgs.write_parquet(target_path)

    return msgs


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


### OPTUNA OPTIMIZATIONS


def suggest_model_settings(trial: optuna.Trial) -> dict:
    # model_type = trial.suggest_categorical("model_type", ["logr", "svc"])
    model_settings = dict(
        model_type="svc",
        C=trial.suggest_float("C", 1e-3, 1e3, log=True),
        max_iter=trial.suggest_int("max_iter", 10, 1000, log=True),
    )
    if model_settings["model_type"] == "svc":
        model_settings["loss"] = trial.suggest_categorical(
            "loss", ["hinge", "squared_hinge"]
        )
        if model_settings["loss"] == "squared_hinge":
            model_settings["penalty"] = trial.suggest_categorical(
                "penalty", ("l1", "l2")
            )
    else:
        model_settings["l1_ratio"] = (
            trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.25),
        )
    return model_settings


def suggest_chat_vec_settings(trial: optuna.Trial) -> dict:
    return {
        "model": trial.suggest_categorical("model", ["tfidf", "counter"]),
        # "max_df": trial.suggest_float("max_df", 0.5, 1.0),
        "ngram_range": (1, trial.suggest_int("ngram_max", 1, 2)),
        "min_df": trial.suggest_int("min_df", 25, 250, step=25),
        "max_features": trial.suggest_int(
            "max_features",
            50,
            1000,
            log=True,
        ),
    }


def gen_objective(X: pl.DataFrame, y: pl.Series):
    @ignore_warnings(category=ConvergenceWarning)
    def objective(trial: optuna.Trial) -> float:
        model_settings = suggest_model_settings(trial)
        model_type = model_settings.pop("model_type")
        match model_type:
            case "logr":
                model = LogisticRegression(**model_settings)
            case "svc":
                if model_settings["loss"] == "hinge":
                    model_settings["max_iter"] *= 2
                model = LinearSVC(**model_settings)
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
    @ignore_warnings(category=ConvergenceWarning)
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
        offset = len(next(cv_time.split(X))[0])
        y_hat = pl.zeros(len(y), dtype=pl.Float64, eager=True)
        for _ in range(max_iter):
            train_inds = [
                sklearn.utils.resample(
                    tr,
                    replace=False,
                    n_samples=len(tr),
                    random_state=trial.number + 123,
                )
                for tr, _ in cv_time.split(X)
            ]
            for batches in zip(
                *(
                    sklearn.utils.gen_batches(len(tr), -(-len(tr) // n_batches))
                    for tr in train_inds
                )
            ):
                # # this is actually much slower becase it pickles stuff.. so need to have a proper pipeline or something better.
                # Xy = zip(*((X[ind := tr[b]], y[ind]) for tr, b in zip(train_inds, batches)))
                # with ProcessPoolExecutor(max_workers=n_splits) as executor:
                #     models = list(executor.map(job_func_full, zip(models, *Xy)))

                for model, (X_, y_) in zip(
                    models,
                    ((X[ind := tr[b]], y[ind]) for tr, b in zip(train_inds, batches)),
                ):
                    model.partial_fit(X_, y_, classes=[False, True])

                for model, (_, ts) in zip(models, cv_time.split(X)):
                    y_hat[ts] = model.decision_function(X[ts])

                g = gini(y[offset:], y_hat[offset:])
                print(f"beginning step: {step}, gini: {g}")
                step += 1
                trial.report(g, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        return g

    return objective_sgd


def learn_chat_embedding(msgs, settings={}, verbose=False):
    # this would take care of sparse matrices
    chat_vec = (
        TfidfVectorizer if settings.get("model", "") == "tfidf" else CountVectorizer
    )
    chat_vec = chat_vec(**{k: v for k, v in settings.items() if k != "model"})
    chat_vec.fit(
        pl.concat(
            [
                msgs.filter(~pl.col("radiant_chat_norm").is_null()).select(
                    chat_norm=pl.col("radiant_chat_norm").list.join(separator=" ")
                ),
                msgs.filter(~pl.col("dire_chat_norm").is_null()).select(
                    chat_norm=pl.col("dire_chat_norm").list.join(separator=" ")
                ),
            ]
        )["chat_norm"].to_list()
    )
    if verbose:
        print("Finished training chat model")
    return chat_vec


def adjoin_chat(Xmsg, chat_vec, mode="both"):
    # this doesn't take advantage of sparce matrices, could be done..
    radiant = pl.from_numpy(
        chat_vec.transform(
            Xmsg.select(
                pl.col("radiant_chat_norm").list.join(separator=" ").fill_null("")
            )
            .to_series()
            .to_list()
        ).toarray(),
        schema=[f"radiant_chat_{x}" for x in chat_vec.get_feature_names_out()],
    )
    dire = pl.from_numpy(
        chat_vec.transform(
            Xmsg.select(pl.col("dire_chat_norm").list.join(separator=" ").fill_null(""))
            .to_series()
            .to_list()
        ).toarray(),
        schema=[f"dire_chat_{x}" for x in chat_vec.get_feature_names_out()],
    )

    if mode == "both":
        both = pl.from_numpy(
            radiant.to_numpy() - dire.to_numpy(),
            schema=[f"chat_{x}" for x in chat_vec.get_feature_names_out()],
        )

        return pl.concat(
            [Xmsg.drop("radiant_chat_norm", "dire_chat_norm"), both], how="horizontal"
        )
    else:
        return pl.concat([Xmsg, radiant, dire], how="horizontal")


def gen_objective_w_chat(X: pl.DataFrame, y: pl.DataFrame):
    @ignore_warnings(category=ConvergenceWarning)
    def objective_w_chat(trial: optuna.Trial) -> float:
        model_settings = suggest_model_settings(trial)
        model_type = model_settings.pop("model_type")
        match model_type:
            case "logr":
                model = LogisticRegression(**model_settings)
            case "svc":
                if model_settings["loss"] == "hinge":
                    model_settings["max_iter"] *= 2
                model = LinearSVC(**model_settings)
            case _:
                raise NotImplementedError

        # HERE INNER VS LEFT IS CRITICAL
        tokenizer_name = trial.suggest_categorical(
            "tokenizer_name", ("toktok", "nist", "destructor", "tweet")
        )
        msgs = get_tokenized_chats(tokenizer_name)
        X_ = (
            X.select("match_id")
            .with_columns(radiant_win=y)
            .join(msgs, on="match_id", how="inner")
            .with_columns(
                pl.col("radiant_chat_norm").fill_null(pl.lit([])),
                pl.col("dire_chat_norm").fill_null(pl.lit([])),
            )
        )
        y_ = X_.get_column("radiant_win")
        X_ = X_.drop("radiant_win")

        chat_settings = suggest_chat_vec_settings(trial)
        chat_vec = learn_chat_embedding(msgs, chat_settings)

        cv_time = TimeSeriesSplit(n_splits=4)
        y_hat = np.concatenate(
            [
                model.fit(
                    adjoin_chat(X_[tr], chat_vec, mode="both").drop("match_id"),
                    y_[tr],
                ).decision_function(
                    adjoin_chat(X_[ts], chat_vec, mode="both").drop("match_id")
                )
                for tr, ts in cv_time.split(X_)
            ]
        )

        return gini(y_[-len(y_hat) :], y_hat)

    return objective_w_chat


def get_objective_sgd_w_chat(X: pl.DataFrame, y: pl.DataFrame):
    @ignore_warnings(category=ConvergenceWarning)
    def objective_sgd_w_chat(trial: optuna.Trial) -> float:
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

        # HERE INNER VS LEFT IS CRITICAL
        tokenizer_name = trial.suggest_categorical(
            "tokenizer_name", ("toktok", "nist", "destructor", "tweet")
        )
        msgs = get_tokenized_chats(tokenizer_name)
        X_ = (
            X.select("match_id")
            .with_columns(radiant_win=y)
            .join(msgs, on="match_id", how="inner")
            .with_columns(
                pl.col("radiant_chat_norm").fill_null(pl.lit([])),
                pl.col("dire_chat_norm").fill_null(pl.lit([])),
            )
        )
        y_ = X_.get_column("radiant_win")
        X_ = X_.drop("radiant_win")

        chat_settings = suggest_chat_vec_settings(trial)
        chat_vec = learn_chat_embedding(msgs, chat_settings)

        max_iter = 8
        n_batches = 1  # make as few batches as possible -- as long as it fits in memory, this speeds up the process
        step = 0
        offset = len(next(cv_time.split(X_))[0])
        y_hat = pl.zeros(len(y_), dtype=pl.Float64, eager=True)
        for _ in range(max_iter):
            train_inds = [
                sklearn.utils.resample(
                    tr,
                    replace=False,
                    n_samples=len(tr),
                    random_state=trial.number + 123,
                )
                for tr, _ in cv_time.split(X_)
            ]
            for batches in zip(
                *(
                    sklearn.utils.gen_batches(len(tr), -(-len(tr) // n_batches))
                    for tr in train_inds
                )
            ):

                for model, (X_batch, y_batch) in zip(
                    models,
                    ((X_[ind := tr[b]], y_[ind]) for tr, b in zip(train_inds, batches)),
                ):

                    model.partial_fit(
                        adjoin_chat(X_batch, chat_vec, mode="both").drop("match_id"),
                        y_batch,
                        classes=[False, True],
                    )

                for model, (_, ts) in zip(models, cv_time.split(X_)):
                    y_hat[ts] = model.decision_function(
                        adjoin_chat(X_[ts], chat_vec, mode="both").drop("match_id")
                    )

                g = gini(y_[offset:], y_hat[offset:])
                print(f"beginning step: {step}, gini: {g}")
                step += 1
                trial.report(g, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return g

    return objective_sgd_w_chat


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


if __name__ == "__main__":
    print("Job started")
    df_train, _ = process_match_df()

    objective_w_chat = gen_objective_w_chat(
        df_train.select("match_id", "radiant_win").collect(),
        df_train.select("radiant_win").collect().to_series(),
    )

    study_name = "chat_study"
    storage = f"sqlite:////data/{study_name}.db"
    sampler = optuna.samplers.TPESampler(seed=10)

    ## if reset
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except KeyError:
        pass  # Study didn't exist yet, which is fine

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction="maximize",
        pruner=None,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective_w_chat, show_progress_bar=True, n_trials=20)

    print(study.best_params)
