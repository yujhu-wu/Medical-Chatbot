from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
import pandas as pd
import numpy as np
import torch
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# ==== 1. 準備資料 ====
with open("intent_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 建 DataFrame 與 Dataset
import pandas as pd

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Label 編碼
label2id = {label: i for i, label in enumerate(sorted(set(df["label"])))}
id2label = {i: label for label, i in label2id.items()}
dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})

# 載入 tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# Tokenize
# Tokenize
def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_fn)

# 切分資料集
# 切分訓練與驗證集
split = tokenized_dataset.train_test_split(test_size=0.2)
train_ds = split["train"]
val_ds = split["test"]

# ==== 2. 載入模型 ====
model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 計算指標函式
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ==== 3. 設定訓練參數 ====
training_args = TrainingArguments(
    output_dir="./intent_model2",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=3e-5,
    warmup_ratio=0.2,
    weight_decay=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    save_total_limit=1
)

# ==== 4. Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ==== 5. 開始訓練並記錄 ====
train_result = trainer.train()

# ==== 6. 儲存模型與 tokenizer ====
trainer.save_model("./intent_model2")
tokenizer.save_pretrained("./intent_model2")

# ==== 7. 擷取訓練記錄並繪圖 ====
log_history = trainer.state.log_history
df_logs = pd.DataFrame(log_history)

# 分離 train loss 和 eval 資訊
df_train = df_logs[df_logs["loss"].notna()][["epoch", "loss"]]

df_eval = df_logs[df_logs["eval_loss"].notna()][
    ["epoch", "eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_f1"]
]

# 把帶 eval_ 的欄位重命名成你原本想用的精簡名稱
df_eval = df_eval.rename(columns={
    "eval_accuracy": "accuracy",
    "eval_precision": "precision",
    "eval_recall":    "recall",
    "eval_f1":        "f1"
})

# Loss 折線圖
plt.figure()
plt.plot(df_train["epoch"], df_train["loss"], label="train_loss")
plt.plot(df_eval["epoch"], df_eval["eval_loss"], label="eval_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Eval Loss")
plt.savefig("loss_per_epoch.png")
plt.close()

# Metrics 折線圖 (Accuracy, Precision, Recall, F1)
plt.figure()
plt.plot(df_eval["epoch"], df_eval["accuracy"], label="accuracy")
plt.plot(df_eval["epoch"], df_eval["precision"], label="precision")
plt.plot(df_eval["epoch"], df_eval["recall"], label="recall")
plt.plot(df_eval["epoch"], df_eval["f1"], label="f1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.title("Evaluation Metrics per Epoch")
plt.savefig("metrics_per_epoch.png")
plt.close()

# ==== 8. 輸出訓練過程 ====
print("=== Training and Evaluation Metrics per Epoch ===")
print("Epoch\tEval Loss\tAccuracy\tPrecision\tRecall\tF1")
for _, row in df_eval.iterrows():
    print(f"{int(row['epoch'])}\t{row['eval_loss']:.4f}\t{row['accuracy']:.4f}\t{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f1']:.4f}")

print("模型已儲存至 ./intent_model2，圖檔 loss_per_epoch.png & metrics_per_epoch.png 已輸出。")
