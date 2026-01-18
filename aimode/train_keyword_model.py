# -*- coding: utf-8 -*-
"""
在 HuggingFace midas/inspec (extraction) 数据集上训练一个
「BERT + TokenClassification (BIO 标签)」的小模型，用来做关键词抽取。

⚠️ 不使用 transformers.Trainer，改为纯 PyTorch 训练循环，
避免导入 transformers.data.metrics → scipy 造成的环境问题。
"""

import os
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

from config import (
    BASE_MODEL_NAME,
    MODEL_DIR,
    MAX_SEQ_LEN,
    EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

# BIO 标签集合
LABEL_LIST = ["O", "B", "I"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# 可选的 seqeval 指标
try:
    from seqeval.metrics import precision_score, recall_score, f1_score

    USE_SEQEVAL = True
except ImportError:
    USE_SEQEVAL = False
    print("⚠️ 没有安装 seqeval，将不在训练中计算 F1（可 pip install seqeval）")


def load_inspec_extraction():
    """
    加载 midas/inspec 的 extraction 配置。
    """
    dataset = load_dataset("midas/inspec", "extraction")
    return dataset["train"], dataset["validation"], dataset["test"]


def tokenize_and_align_labels(example: Dict[str, Any], tokenizer):
    """
    把 Inspec 的 [word 列表] + [BIO 标签] 转成适合 BERT 的输入：
    - is_split_into_words=True
    - 对 subword 只给第一个 subword 标注，其余设为 -100（不算 loss）
    - 直接 padding='max_length'，这样 DataLoader 可以用默认 collate_fn
    """
    words = example["document"]          # list[str]
    tags = example["doc_bio_tags"]       # list[str]，元素是 "B"/"I"/"O"

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_offsets_mapping=False,
    )

    word_ids = encoding.word_ids()
    label_ids = []

    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            label_ids.append(-100)
        elif word_id != previous_word_id:
            tag_str = tags[word_id]
            label_ids.append(LABEL2ID[tag_str])
        else:
            label_ids.append(-100)
        previous_word_id = word_id

    encoding["labels"] = label_ids
    return encoding


def prepare_datasets(tokenizer):
    """对数据集做预处理，并转成可以直接给 DataLoader 用的形式。"""
    train_ds, val_ds, test_ds = load_inspec_extraction()

    def _preprocess(examples):
        return tokenize_and_align_labels(examples, tokenizer)

    encoded_train = train_ds.map(_preprocess, batched=False)
    encoded_val = val_ds.map(_preprocess, batched=False)
    encoded_test = test_ds.map(_preprocess, batched=False)

    # 去掉原始列，只保留 input_ids, attention_mask, labels
    keep_cols = ["input_ids", "attention_mask", "labels"]
    encoded_train = encoded_train.remove_columns(
        [c for c in encoded_train.column_names if c not in keep_cols]
    )
    encoded_val = encoded_val.remove_columns(
        [c for c in encoded_val.column_names if c not in keep_cols]
    )
    encoded_test = encoded_test.remove_columns(
        [c for c in encoded_test.column_names if c not in keep_cols]
    )

    # 让 datasets 返回 torch.Tensor
    encoded_train.set_format(type="torch", columns=keep_cols)
    encoded_val.set_format(type="torch", columns=keep_cols)
    encoded_test.set_format(type="torch", columns=keep_cols)

    return encoded_train, encoded_val, encoded_test


def make_dataloaders(train_ds, val_ds, test_ds) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def decode_preds_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    id2label: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    把一个 batch 的 logits/labels 转成 seqeval 需要的标签序列。
    """
    preds = logits.argmax(-1).cpu().numpy()
    labels = labels.cpu().numpy()

    batch_true = []
    batch_pred = []

    for pred_ids, label_ids in zip(preds, labels):
        true_tags = []
        pred_tags = []
        for p, l in zip(pred_ids, label_ids):
            if l == -100:
                continue
            true_tags.append(id2label[int(l)])
            pred_tags.append(id2label[int(p)])
        batch_true.append(true_tags)
        batch_pred.append(pred_tags)

    return batch_true, batch_pred


def evaluate(
    model,
    data_loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
) -> Dict[str, float]:
    """在 val/test 上跑一遍，返回 loss 和（如果有的话）F1 等指标。"""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    all_true: List[List[str]] = []
    all_pred: List[List[str]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Eval", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1

            if USE_SEQEVAL:
                bt_true, bt_pred = decode_preds_and_labels(
                    outputs.logits, batch["labels"], id2label
                )
                all_true.extend(bt_true)
                all_pred.extend(bt_pred)

    avg_loss = total_loss / max(total_steps, 1)

    metrics = {"loss": avg_loss}
    if USE_SEQEVAL and all_true:
        p = precision_score(all_true, all_pred)
        r = recall_score(all_true, all_pred)
        f = f1_score(all_true, all_pred)
        metrics.update(
            {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
            }
        )

    return metrics


def train():
    """
    在 Inspec 数据集上训练，并保存到 MODEL_DIR。
    """
    print("📚 Loading tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    train_ds, val_ds, test_ds = prepare_datasets(tokenizer)
    print(f"Train size: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🖥  Using device:", device)

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_f1 = -1.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        # ------- 训练 -------
        model.train()
        total_loss = 0.0
        total_steps = 0

        for batch in tqdm(train_loader, desc="Train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        avg_train_loss = total_loss / max(total_steps, 1)
        print(f"  Train loss: {avg_train_loss:.4f}")

        # ------- 验证 -------
        val_metrics = evaluate(model, val_loader, device, ID2LABEL)
        print(f"  Val loss: {val_metrics['loss']:.4f}")
        if USE_SEQEVAL and "f1" in val_metrics:
            print(
                f"  Val F1: {val_metrics['f1']:.4f}, "
                f"P: {val_metrics['precision']:.4f}, "
                f"R: {val_metrics['recall']:.4f}"
            )

            # 保存最好的模型
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                print(f"  ✅ New best F1: {best_f1:.4f}, saving model to {MODEL_DIR}")
                model.save_pretrained(MODEL_DIR)
                tokenizer.save_pretrained(MODEL_DIR)
        else:
            # 没有 seqeval 就按 loss 存一下
            if best_f1 < 0 or val_metrics["loss"] < best_f1:
                best_f1 = val_metrics["loss"]
                print(f"  ✅ New best (by loss): {best_f1:.4f}, saving model to {MODEL_DIR}")
                model.save_pretrained(MODEL_DIR)
                tokenizer.save_pretrained(MODEL_DIR)

    # ------- 在 test 上简单评估一下 -------
    print("\n📊 Evaluating best model on test set...")
    # 重新加载保存好的 best 模型（稳妥一点）
    best_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR
    ).to(device)
    test_metrics = evaluate(best_model, test_loader, device, ID2LABEL)
    print("Test metrics:", test_metrics)

    print("\n✅ 训练完成，最佳模型保存在：", MODEL_DIR)


if __name__ == "__main__":
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        print(f"模型目录 {MODEL_DIR} 已存在，如需重新训练请先手动清空该目录。")
    else:
        train()
