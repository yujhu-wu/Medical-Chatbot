from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
import pandas as pd
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score


with open("intent_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Label 編碼
label2id = {label: i for i, label in enumerate(sorted(set(df["label"])))}
id2label = {i: label for label, i in label2id.items()}
dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

# 載入 tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# Tokenize
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_fn)

# 切分訓練與驗證集
split = tokenized_dataset.train_test_split(test_size=0.2)
train_ds = split["train"]
val_ds = split["test"]

# 載入模型
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir="./intent_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=3e-5,
    warmup_ratio=0.2,
    weight_decay=0.05,

    eval_strategy="epoch",
    save_strategy="epoch",          # 或 "steps"
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    logging_steps=10,
    save_total_limit=1,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 開始訓練
trainer.train()

# 儲存模型
trainer.save_model("./intent_model")
tokenizer.save_pretrained("./intent_model")
