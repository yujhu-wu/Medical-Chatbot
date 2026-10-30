from transformers import BertTokenizer, BertForSequenceClassification
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
import random
import sys, io
import pandas as pd
from collections import defaultdict


# 確保中文能正常輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ========== 載入 intent 分類模型 ==========
intent_model_path = "./intent_model"
intent_tokenizer = BertTokenizer.from_pretrained(intent_model_path)
intent_model     = BertForSequenceClassification.from_pretrained(intent_model_path)
intent_model.eval()
id2label = intent_model.config.id2label  # e.g. {0:'intro',1:'need_doctor',...}

# ========== 載入 回答範本資料集 ==========
# 假設 csv 放在專案根目錄下
templates_df = pd.read_csv("response_templates.csv")

# 建立一個 dict: (disease, intent) → [範本1, 範本2, …]
template_map = defaultdict(list)
for _, row in templates_df.iterrows():
    # input 欄位格式範例: "disease=MEL intent=intro"
    disease_k, intent_k = row["input"].split()
    disease = disease_k.split("=")[1]
    intent  = intent_k.split("=")[1]
    template_map[(disease, intent)].append(row["target"])

# ========== fallback 回應 ==========
fallback_replies = [
    "我聽不太懂你的問題，可以再說一次嗎？",
    "我目前只能回答皮膚病相關的問題，例如『這是什麼？』、『需不需要看醫生？』",
    "請問你想知道這個疾病的哪方面資訊？"
]

# ========== 預測意圖函式（不變） ==========
def predict_intent(user_input: str) -> str:
    inputs = intent_tokenizer(user_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = intent_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    max_p, pid = torch.max(probs, dim=-1)
    if max_p.item() < 0.6:
        return "fallback"
    return id2label[pid.item()]

# ========== 生成回答：直接從範本集中隨機選一條 ==========
# 建立 (disease,intent) → 全部範本 清單
template_map = defaultdict(list)
for _, row in templates_df.iterrows():
    disease_k, intent_k = row["input"].split()
    disease = disease_k.split("=")[1]
    intent  = intent_k.split("=")[1]
    template_map[(disease, intent)].append(row["target"])

# 建立 (disease,intent) → 未用過的範本池（初始複製 template_map）
template_pool_map = { key: val.copy() for key, val in template_map.items() }

def generate_response(disease: str, intent: str) -> str:
    # fallback 邏輯不變
    if intent == "fallback":
        return random.choice(fallback_replies)

    key = (disease, intent)
    all_templates = template_map.get(key, [])
    if not all_templates:
        # 找不到任何範本，回 fallback
        return random.choice(fallback_replies)

    # 如果 pool 裡已經沒有剩餘範本，就重置 pool
    pool = template_pool_map.get(key, [])
    if not pool:
        # 重置為全新的清單（複製一份原本的 list）
        pool = all_templates.copy()
        template_pool_map[key] = pool

    # 隨機從 pool 挑一條，然後從 pool 裡移除
    choice = random.choice(pool)
    pool.remove(choice)

    return choice

if __name__ == "__main__":
    print("智慧醫療對話系統啟動！")
    disease = input("請輸入圖像模型判斷的疾病類別（如 MEL）：").strip().upper()
    while True:
        user_input = input("你：").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("再見～保重身體！")
            break
        intent   = predict_intent(user_input)
        response = generate_response(disease, intent)
        print(f"\n[預測意圖] {intent}")
        print(f"{response}\n")
