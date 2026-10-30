import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration # 確保這裡使用 MT5Tokenizer 和 MT5ForConditionalGeneration
import re
import random
import os

# 疾病文字說明 (用於在訓練時生成輸入，必須與 app.py 中的 info_dict 一致)
info_dict = {
    'MEL': "惡性黑色素瘤",
    'NV': "黑色素痣",
    'BCC': "基底細胞癌",
    'AKIEC': "光化性角化病", # 更正為「光化性角化病」
    'BKL': "脂漏性角化病", # 更正為「脂漏性角化病」
    'DF': "皮膚纖維瘤", # 更正為「皮膚纖維瘤」
    'VASC': "血管病變",
    'UNK7': "未知疾病7", # 請替換為實際的中文名稱
    'UNK8': "未知疾病8",
    'UNK9': "未知疾病9"  # 請替換為實際的中文名稱
}

_fallback_responses = [
    "我聽不太懂你的問題，可以再說一次嗎？",
    "我目前只能回答皮膚病相關的問題，例如『這是什麼？』、『需不需要看醫生？』",
    "請問你想知道這個疾病的哪方面資訊？"
]

class MedicalChatbot:
    def __init__(self, model_path="./medical_chatbot_final"):
        """
        初始化醫療聊天機器人
        """
        print("載入醫療聊天機器人...")
        try:
            # 確保這裡使用的模型和 tokenizer 與訓練時保存的類型一致
            self.tokenizer = MT5Tokenizer.from_pretrained(model_path)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path)

            if torch.cuda.is_available():
                self.model.to("cuda")
                print("推理模型已移至 CUDA 設備。")
            else:
                print("CUDA 不可用，推理將在 CPU 上進行。")

        except Exception as e:
            print(f"嘗試載入 T5/MT5 模型失敗: {e}")
            print("嘗試改用 BART 模型載入 (這通常表示模型路徑或類型不匹配)...")
            # 備用方案，如果您確定只使用 T5/MT5，可以考慮移除，或確保有 BART 模型在該路徑
            from transformers import BartTokenizer, BartForConditionalGeneration
            try:
                self.tokenizer = BartTokenizer.from_pretrained(model_path)
                self.model = BartForConditionalGeneration.from_pretrained(model_path)
                if torch.cuda.is_available():
                    self.model.to("cuda")
                    print("推理模型已移至 CUDA 設備 (BART)。")
                else:
                    print("CUDA 不可用，推理將在 CPU 上進行 (BART)。")
            except Exception as bart_e:
                print(f"嘗試載入 BART 模型也失敗: {bart_e}")
                print("請檢查模型路徑和檔案是否正確，以及模型類型是否與保存的權重匹配。")
                raise  # 重新拋出錯誤，讓應用程式知道模型載入失敗

        self.model.eval() # 設定為評估模式
        print("模型載入完成！")

    def chat(self, question_with_context): # 這裡的參數現在是帶上下文的問題
        """
        與聊天機器人對話。
        question_with_context 應是一個包含疾病資訊的完整輸入字串，
        例如："惡性黑色素瘤 (MEL) 相關問題：這是什麼？"
        """
        input_text = question_with_context # 直接使用傳入的帶上下文的問題
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=128, # 最大輸入長度，根據訓練時的設定
            truncation=True
        )

        # 將 input_ids 也移到模型所在的設備 (GPU 或 CPU)
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # 生成回答
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=256, # 最大輸出長度，根據訓練時的設定
                min_length=10,
                num_beams=1,        # 保持為 1，啟用純採樣 (nucleus/top-k sampling)
                do_sample=True,     # 保持為 True，啟用採樣
                temperature=1.0,    # **提高溫度至 1.0**：最大化隨機性，讓分佈更平坦。
                top_k=50,           # **重新啟用 top_k**：結合 top_p 使用。設置為一個合理的值 (例如 50)，確保模型考慮足夠多的詞彙。
                top_p=0.90,         # **略微降低 top_p**：將累積機率限制在 90%。這有助於避免模型選擇過於不相關或奇怪的詞彙，同時保留較高的隨機性。
                repetition_penalty=2.0,# **進一步提高重複懲罰**：讓模型更積極地避免重複。可以嘗試 2.0 到 2.5 之間。
                no_repeat_ngram_size=3,# 保持 3，防止 3 個詞長的 n-grams 重複。這是一個很有效的防止重複的參數。
                num_return_sequences=1,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 解碼回答
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_response(response) # 使用清理函數

         # 如果模型回傳空或不合理，就選擇一條 fallback 回應
        if not response or response.strip() == "":
            response = random.choice(_fallback_responses)
        return response

    def _clean_response(self, response):
        # 這裡可以添加一些清理邏輯，如果模型回答有不必要的字符
        # 移除可能的特殊 token，例如 <pad>
        return response.replace("<pad>", "").strip()

    def batch_test(self, test_data):
        print("\n=== 批量測試 ===")
        for i, item in enumerate(test_data):
            # 這裡的 item['question'] 應該已經是帶上下文的完整問題
            question = item['question']
            original_target = item.get('original', 'N/A') # 處理可能沒有 original 鍵的情況
            response = self.chat(question)
            print(f"{i+1}. 問: {question}\n    答: {self._clean_response(response)}")
            # 您可以根據需要在此處添加原始目標的顯示
            # print(f"    原始: {original_target}")

    def start_conversation(self):
        print("\n=== 醫療聊天機器人 (測試模式) ===")
        print("您好！我是醫療諮詢助手，在測試模式下，請直接輸入包含疾病和問題的完整句子。")
        print("例如：'惡性黑色素瘤 (MEL) 相關問題：這是什麼？'")
        print("(輸入 'quit' 結束對話)")
        while True:
            user_input = input("\n您: ")
            if user_input.lower() == 'quit':
                break
            # 直接將用戶輸入傳給 chat，因為它應該已經是完整格式
            response = self.chat(user_input)
            print(f"機器人: {self._clean_response(response)}")

# 注意： predict_intent 和 generate_response 函數不再需要在此模組中，
# 因為它們的功能已被 T5/MT5 模型在整個上下文處理中取代。
# 如果它們是其他意圖分類或特定生成邏輯的一部分，則應放在其他相關模組中。
