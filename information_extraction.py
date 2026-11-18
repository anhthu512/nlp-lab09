import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baseline_ner import LAB_DIR, SEED, bio_spans, entity_metrics, load_conll_split

DEFAULT_CHECKPOINT = LAB_DIR / "outputs" / "ner_checkpoint"
ENTITY_KEYS = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"}
DEMO_TEXT = (
    "Barack Obama spoke at Harvard University in Boston. "
    "Bill Gates attended a conference at Stanford University in California."
)


def load_ner_checkpoint(path=DEFAULT_CHECKPOINT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(path).to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def extract_entity_spans(text, tokenizer, model, batch_size=32):
    max_length = model.config.max_position_embeddings
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=64,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    model.eval()
    word_offsets = {}
    word_tags = {}
    for start in range(0, len(encoded["input_ids"]), batch_size):
        stop = start + batch_size
        inputs = {key: encoded[key][start:stop].to(device) for key in tokenizer.model_input_names}
        prediction_ids = model(**inputs).logits.argmax(dim=-1).cpu().tolist()
        for window_index, token_tags in enumerate(prediction_ids, start=start):
            word_ids = encoded.word_ids(window_index)
            offsets = encoded["offset_mapping"][window_index].tolist()
            for word_id, (left, right), tag_id in zip(word_ids, offsets, token_tags, strict=True):
                if word_id is None:
                    continue
                if word_id not in word_offsets:
                    word_offsets[word_id] = [left, right]
                    word_tags[word_id] = model.config.id2label[tag_id]
                else:
                    word_offsets[word_id][1] = max(word_offsets[word_id][1], right)

    ordered_ids = sorted(word_tags)
    tags = [word_tags[idx] for idx in ordered_ids]
    return [
        (word_offsets[ordered_ids[start]][0], word_offsets[ordered_ids[end - 1]][1], entity_type)
        for start, end, entity_type in bio_spans(tags)
        if entity_type in ENTITY_KEYS
    ]


def extract_information(text, tokenizer, model):
    result = {key: [] for key in ENTITY_KEYS.values()}
    for start, end, entity_type in extract_entity_spans(text, tokenizer, model):
        result[ENTITY_KEYS[entity_type]].append(text[start:end])
    return result


def evaluate_extraction(test_rows, label_names, tokenizer, model, sample_size=30):
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(test_rows), size=sample_size, replace=False)
    gold, predicted = set(), set()
    for index in indices.tolist():
        words = test_rows[index]["tokens"]
        tags = [label_names[tag] for tag in test_rows[index]["ner_tags"]]
        starts = np.cumsum([0, *(len(word) + 1 for word in words[:-1])]).tolist()
        gold.update(
            (index, starts[start], starts[end - 1] + len(words[end - 1]), entity_type)
            for start, end, entity_type in bio_spans(tags)
            if entity_type in ENTITY_KEYS
        )
        predicted.update(
            (index, start, end, entity_type)
            for start, end, entity_type in extract_entity_spans(" ".join(words), tokenizer, model)
        )
    return {
        "Sentences": sample_size,
        "Gold": len(gold),
        "Predicted": len(predicted),
        **entity_metrics(gold, predicted),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract PERSON, ORGANIZATION, and LOCATION from raw text.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--text", default=DEMO_TEXT)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on 30 CoNLL-2003 test sentences.")
    args = parser.parse_args()
    if args.checkpoint == DEFAULT_CHECKPOINT and not DEFAULT_CHECKPOINT.is_dir():
        parser.error("Run ner_finetuning.ipynb to create the fine-tuned checkpoint.")
    tokenizer, model = load_ner_checkpoint(args.checkpoint)
    print(json.dumps(extract_information(args.text, tokenizer, model), ensure_ascii=False, indent=2))

    if args.evaluate:
        test_table = load_conll_split("test")
        features = json.loads(test_table.schema.metadata[b"huggingface"])["info"]["features"]
        label_names = features["ner_tags"]["feature"]["names"]
        metrics = evaluate_extraction(test_table.to_pylist(), label_names, tokenizer, model)
        print("\nCoNLL-2003 test subset | Exact character span and type match")
        print(pd.DataFrame([metrics]).to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
