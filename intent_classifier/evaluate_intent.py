# evaluate_and_plot.py

import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# ==== 1. 參數設定 ====
MODEL_DIR        = "./intent_model"                  # 你的已存模型資料夾
DATA_JSON        = "intent_dataset.json"   # 你的驗證/測試 JSON 檔
BATCH_SIZE       = 32
PLOT_OUTPUT      = "evaluation_metrics.png"

# ==== 2. 載入模型 & tokenizer ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model     = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ==== 3. 載入並準備資料 ====
with open(DATA_JSON, "r", encoding="utf-8") as f:
    raw = json.load(f)
df = pd.DataFrame(raw)

# 如果你有做 label2id / id2label mapping，請一併重建：
labels = sorted(df["label"].unique())
label2id = {lab:i for i,lab in enumerate(labels)}
df["label_id"] = df["label"].map(label2id)

# tokenize
enc = tokenizer(
    df["text"].tolist(),
    padding=True, truncation=True,
    return_tensors="pt"
)

dataset = TensorDataset(
    enc["input_ids"],
    enc["attention_mask"],
    torch.tensor(df["label_id"].tolist())
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ==== 4. 推理 & 收集預測結果 ====
all_preds = []
all_labels = []

with torch.no_grad():
    for input_ids, attention_mask, lbl in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(lbl.tolist())

# ==== 5. 計算指標 ====
acc = accuracy_score(all_labels, all_preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted"
)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# ==== 6. 畫圖並存檔 ====
metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1
}

plt.figure()
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1)
plt.title("Evaluation Metrics")
for i,(k,v) in enumerate(metrics.items()):
    plt.text(i, v+0.02, f"{v:.2f}", ha="center")
plt.savefig(PLOT_OUTPUT, bbox_inches="tight")
plt.close()

print(f"評分圖已存檔為：{PLOT_OUTPUT}")
