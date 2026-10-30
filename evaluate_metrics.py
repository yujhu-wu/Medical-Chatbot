import math
import pandas as pd
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import evaluate
from datasets import Dataset
import torch
import jieba

# 載入 tokenizer 和模型
model_path = "./medical_chatbot_final"
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 載入測試集
df = pd.read_csv("natural_medical_qa.csv")  # 確保這是你訓練時產生的資料
df = df[['input', 'target']].dropna().copy()
df = df.sample(frac=1.0).reset_index(drop=True)  # 打亂

# 只取一部分測試（加速評估，可改）
df = df[:300]

dataset = Dataset.from_pandas(df)

# 載入指標
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# 儲存預測與標準答案
preds, labels = [], []
losses = []

# 評估
print("開始推理與評估...")

for example in dataset:
    input_text = example['input']
    target_text = example['target']

    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    labels_ids = tokenizer.encode(target_text, return_tensors="pt", truncation=True, max_length=256).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=256,
            num_beams=5,
            do_sample=False,
            early_stopping=True
        )
        logits = model(input_ids=inputs, labels=labels_ids).loss
        losses.append(logits.item())

    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    preds.append(pred.strip())
    labels.append(target_text.strip())

# === 1. Perplexity ===
avg_loss = sum(losses) / len(losses)
perplexity = math.exp(avg_loss)

# === 2. ROUGE ===
rouge_result = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
rouge_l = rouge_result["rougeL"] * 100

# === 3. BLEU ===
bleu_preds = [' '.join(jieba.lcut(p)) for p in preds]
bleu_refs = [' '.join(jieba.lcut(l)) for l in labels]
bleu_score = bleu.compute(predictions=bleu_preds, references=bleu_refs)['bleu'] * 100

# === 顯示結果 ===
print("\n===== 評估結果 =====")
print(f"Perplexity: {perplexity:.2f}")
print(f"ROUGE-L:    {rouge_l:.2f}")
print(f"BLEU:       {bleu_score:.2f}")

# === 顯示部分樣本 ===
print("\n=== 範例輸出 (前5筆) ===")
for i in range(5):
    print(f"問題: {df['input'][i]}")
    print(f"標準答案: {df['target'][i]}")
    print(f"模型回答: {preds[i]}")
    print("-----")
