import json
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

LAB_DIR = Path(__file__).resolve().parent
SEED = 42
DATASET_ID = "eriktks/conll2003"
DATASET_REVISION = "e8237f8875e4ecf7f60b07ae49691b85cf893410"


def load_conll_split(split):
    path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=f"conll2003/conll2003-{split}.parquet",
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    return pq.read_table(path, columns=["tokens", "ner_tags"])


def load_conll_data():
    train_table = load_conll_split("train")
    test_table = load_conll_split("test")
    features = json.loads(train_table.schema.metadata[b"huggingface"])["info"]["features"]
    label_names = features["ner_tags"]["feature"]["names"]
    return train_table.to_pylist(), test_table.to_pylist(), label_names


def token_features(words):
    lower_words = [word.lower() for word in words]
    for i, (word, lower) in enumerate(zip(words, lower_words)):
        yield {
            "word": lower,
            "prefix2": lower[:2],
            "prefix3": lower[:3],
            "suffix2": lower[-2:],
            "suffix3": lower[-3:],
            "is_title": word.istitle(),
            "is_upper": word.isupper(),
            "is_lower": word.islower(),
            "is_digit": word.isdigit(),
            "has_digit": any(char.isdigit() for char in word),
            "has_hyphen": "-" in word,
            "previous": lower_words[i - 1] if i else "<BOS>",
            "next": lower_words[i + 1] if i + 1 < len(words) else "<EOS>",
        }


def fit_baseline(rows):
    model = make_pipeline(
        DictVectorizer(),
        SGDClassifier(loss="log_loss", alpha=1e-5, average=True, max_iter=50, random_state=SEED),
    )
    features = chain.from_iterable(token_features(row["tokens"]) for row in rows)
    labels = list(chain.from_iterable(row["ner_tags"] for row in rows))
    return model.fit(features, labels)


def predict_baseline(model, sentences, label_names):
    features = chain.from_iterable(map(token_features, sentences))
    flat_predictions = model.predict(features)
    boundaries = np.cumsum([len(words) for words in sentences[:-1]], dtype=int)
    return [
        [label_names[tag] for tag in sentence]
        for sentence in np.split(flat_predictions, boundaries)
    ]


def bio_spans(tags):
    active_type = None
    for end, tag in enumerate(chain(tags, ["O"])):
        prefix, _, entity_type = tag.partition("-")
        continues = active_type is not None and prefix == "I" and entity_type == active_type
        if continues:
            continue
        if active_type is not None:
            yield start, end, active_type
        active_type = entity_type if prefix in {"B", "I"} else None
        start = end


def corpus_spans(tag_sequences):
    return {
        (sentence_id, start, end, entity_type)
        for sentence_id, tags in enumerate(tag_sequences)
        for start, end, entity_type in bio_spans(tags)
    }


def entity_metrics(gold, predicted):
    true_positives = len(gold & predicted)
    return {
        "Precision": true_positives / len(predicted) if predicted else 0.0,
        "Recall": true_positives / len(gold) if gold else 0.0,
        "F1": 2 * true_positives / (len(gold) + len(predicted)) if gold or predicted else 0.0,
    }


def main():
    train, test, label_names = load_conll_data()
    model = fit_baseline(train)
    test_words = [row["tokens"] for row in test]
    predicted = predict_baseline(model, test_words, label_names)
    gold = [[label_names[tag] for tag in row["ner_tags"]] for row in test]
    metrics = entity_metrics(corpus_spans(gold), corpus_spans(predicted))
    print(f"Train = {len(train):,} sentences | Test = {len(test):,} sentences")
    print(pd.DataFrame([{"Model": "Baseline", **metrics}]).to_string(index=False, float_format="{:.4f}".format))
    print()
    print(pd.DataFrame({"Word": test_words[0], "Gold": gold[0], "Prediction": predicted[0]}).to_string(index=False))


if __name__ == "__main__":
    main()
