import pandas as pd
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import torch
import re
import random
import os

# 警告處理 (保持不變)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The tokenizer class you load from this checkpoint is not the same type as the class this function is called from.*")
warnings.filterwarnings("ignore", category=UserWarning, message="You are using a model of type t5 to instantiate a model of type mt5. This is not supported for all configurations of models and can yield errors.*")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future.*")


# ===== 疾病中英文名稱對應字典 (新增/確保存在並完整) =====
# 這個字典對於構建統一的訓練輸入格式至關重要
# 請根據你的實際數據進行補充
disease_chinese_name_map = {
    "BCC": "基底細胞癌",
    "BKL": "脂漏性角化病",
    "NV": "黑色素痣",
    "AKIEC": "光化性角化病",
    "DF": "纖維瘤",
    "MEL": "惡性黑色素瘤",
    "VASC": "血管瘤",
    # 請在此處新增其他疾病的對應，例如：
    # "SCC": "鱗狀細胞癌",
    # "HEM": "血管瘤",
}

# ===== 第一步：從 CSV 讀取並轉換資料 =====

def parse_structured_input(structured_input):
    """
    解析結構化輸入：disease=BCC intent=treatment
    """
    disease_match = re.search(r'disease=(\w+)', structured_input)
    intent_match = re.search(r'intent=(\w+)', structured_input)

    disease = disease_match.group(1) if disease_match else "未知疾病"
    intent = intent_match.group(1) if intent_match else "unknown"

    return disease, intent

def generate_natural_questions(disease_abbr, intent): # 將參數名改為 disease_abbr 更清晰
    """
    根據疾病縮寫和意圖生成自然語言問題
    """
    # 這裡的模板直接使用 disease_abbr，因為它會被用於構建完整的輸入字符串
    question_templates = {
        'treatment': [
            f"{disease_abbr} 要怎麼治療？",
            f"{disease_abbr} 的治療方法是什麼？",
            f"我得了 {disease_abbr}，該怎麼辦？",
            f"{disease_abbr} 如何處理？",
            f"請問 {disease_abbr} 要如何治療？",
            f"{disease_abbr} 可以怎麼治療？",
            f"請問 {disease_abbr} 要怎麼治療比較好？",
            f"得了 {disease_abbr} 的話，有哪些治療方法可以選擇？",
            f"針對 {disease_abbr} 這個病，建議採取什麼治療方案呢？",
            f"{disease_abbr} 有沒有什麼藥物或手術可以處理？",
            f"如果我被診斷出 {disease_abbr}，下一步該怎麼辦？",
            f"{disease_abbr} 的治療流程是怎樣的？",
            f"{disease_abbr} 的治療方案有哪些？",
            f"我該如何治療 {disease_abbr}？",
            f"{disease_abbr} 可以不用治療嗎？",
            f"{disease_abbr} 的治療費用高嗎？",
            f"{disease_abbr} 治療後會復發嗎？",
            f"針對 {disease_abbr} 有沒有非手術的治療選項？",
            f"治療 {disease_abbr} 的最新方法是什麼？",
            f"{disease_abbr} 的居家護理方法有哪些？",
            f"治療 {disease_abbr} 的成功率大概是多少？",
            f"我適合哪種 {disease_abbr} 的治療方式？",
            f" {disease_abbr} 治療期間有什麼需要注意的嗎？",
            f"有什麼方法可以緩解 {disease_abbr} 的症狀？",
            f" {disease_abbr} 的中醫治療方法是什麼？",
            f" {disease_abbr} 可以用雷射治療嗎？",
            f"針對 {disease_abbr}，醫生通常會建議什麼治療？",
            f"我應該在什麼時候開始治療 {disease_abbr}？"
        ],
        'contagious': [
            f"{disease_abbr} 會傳染嗎？",
            f"{disease_abbr} 會不會傳染給別人？",
            f"我的 {disease_abbr} 會傳染嗎？",
            f"{disease_abbr} 具有傳染性嗎？",
            f"{disease_abbr} 會不會感染他人？",
            f"{disease_abbr} 會傳播嗎？",
            f"{disease_abbr} 會不會傳染給家人或朋友？",
            f"這種 {disease_abbr} 是傳染性的嗎？",
            f"我得了 {disease_abbr}，會不會感染其他人？",
            f"{disease_abbr} 是否有傳播的風險？",
            f"關於 {disease_abbr} 的傳染性問題，我需要注意什麼？",
            f"{disease_abbr} 是由接觸傳播的嗎？",
            f"{disease_abbr} 會不會交叉感染？",
            f"{disease_abbr} 有無傳染性？",
            f"{disease_abbr} 的傳染途徑是什麼？",
            f"{disease_abbr} 是通過什麼方式傳染的？",
            f"如果我接觸了 {disease_abbr} 患者，會被傳染嗎？",
            f"{disease_abbr} 在什麼情況下會傳染？",
            f"怎麼預防 {disease_abbr} 傳染？",
            f"{disease_abbr} 會通過空氣傳染嗎？",
            f"{disease_abbr} 會通過血液傳染嗎？",
            f" {disease_abbr} 會從母體傳染給胎兒嗎？",
            f"寵物會把 {disease_abbr} 傳染給我嗎？",
            f" {disease_abbr} 傳染期是多久？"
        ],
        'intro': [
            f"什麼是 {disease_abbr}？",
            f"{disease_abbr} 是什麼病？",
            f"請介紹一下 {disease_abbr}",
            f"{disease_abbr} 是怎樣的疾病？",
            f"能解釋一下 {disease_abbr} 嗎？",
            f"{disease_abbr} 是什麼？",
            f"您可以簡要介紹一下 {disease_abbr} 是什麼嗎？",
            f"我想了解 {disease_abbr} 這個疾病，它主要表現是什麼？",
            f"能不能給我解釋一下 {disease_abbr} 的基本資訊？",
            f"{disease_abbr} 是一種什麼樣的皮膚狀況？",
            f"請說明一下 {disease_abbr} 的特點。",
            f"{disease_abbr} 的定義是什麼？",
            f"{disease_abbr} 是什麼病症？",
            f"介紹一下 {disease_abbr}。",
            f"{disease_abbr} 的成因是什麼？",
            f"{disease_abbr} 的症狀有哪些？",
            f"{disease_abbr} 通常出現在哪個部位？",
            f"{disease_abbr} 和其他皮膚病有什麼區別？",
            f" {disease_abbr} 的病理學特點是什麼？",
            f"{disease_abbr} 的發病率高嗎？",
            f" {disease_abbr} 有哪些亞型？",
            f" {disease_abbr} 的臨床表現是怎樣的？",
            f"這種 {disease_abbr} 是怎麼形成的？",
            f" {disease_abbr} 的診斷標準是什麼？"
        ],
        'risk': [
            f"{disease_abbr} 會惡化嗎？",
            f"{disease_abbr} 有什麼風險？",
            f"{disease_abbr} 危險嗎？",
            f"{disease_abbr} 會變嚴重嗎？",
            f"{disease_abbr} 需要擔心嗎？",
            f"{disease_abbr} 會有併發症嗎？",
            f"{disease_abbr} 會不會惡化成更嚴重的疾病？",
            f"{disease_abbr} 有哪些潛在的健康風險或併發症？",
            f"患有 {disease_abbr} 需要擔心癌變嗎？",
            f"{disease_abbr} 的長期預後如何？會不會有危險？",
            f"如果我患有 {disease_abbr}，需要警惕哪些風險？",
            f"{disease_abbr} 會對身體造成什麼影響？",
            f"{disease_abbr} 的危險性高嗎？",
            f"{disease_abbr} 有什麼潛在問題？",
            f"{disease_abbr} 會影響生活品質嗎？",
            f"{disease_abbr} 會不會危及生命？",
            f" {disease_abbr} 有什麼長期後遺症嗎？",
            f"得了 {disease_abbr} 會有哪些不好的影響？",
            f"{disease_abbr} 會不會留下疤痕？",
            f"{disease_abbr} 和其他疾病有相關性嗎？",
            f" {disease_abbr} 的復發機率大嗎？",
            f" {disease_abbr} 會引起疼痛嗎？",
            f" {disease_abbr} 的致死率高嗎？",
            f"有哪些情況下 {disease_abbr} 會變得更嚴重？"
        ],
        'need_doctor': [
            f"{disease_abbr} 需要看醫生嗎？",
            f"得了 {disease_abbr} 要去醫院嗎？",
            f"{disease_abbr} 什麼時候要就診？",
            f"{disease_abbr} 需要就醫嗎？",
            f"有 {disease_abbr} 要看醫生嗎？",
            f"{disease_abbr} 要不要去看醫生？",
            f"得了 {disease_abbr} 什麼時候需要立刻去看醫生？",
            f"如果我被診斷出 {disease_abbr}，是否需要馬上就醫？",
            f"關於 {disease_abbr}，有哪些情況是必須尋求醫生幫助的？",
            f"{disease_abbr} 這個狀況，我應該去看哪個科別的醫生？",
            f"患有 {disease_abbr} 的話，有沒有必要去醫院檢查一下？",
            f"我應該在什麼情況下為 {disease_abbr} 就診？",
            f"什麼時候需要因為 {disease_abbr} 去醫院？",
            f"{disease_abbr} 怎樣才需要看醫生？",
            f"{disease_abbr} 可以自行處理嗎？",
            f"什麼時候我需要緊急就醫 {disease_abbr}？",
            f"得了 {disease_abbr} 要掛什麼科？",
            f" {disease_abbr} 看醫生前需要準備什麼？",
            f" {disease_abbr} 可以在一般診所看嗎？還是要去大醫院？",
            f" {disease_abbr} 需要定期回診嗎？",
            f"如果 {disease_abbr} 惡化了，要立即就醫嗎？",
            f"醫生會如何診斷 {disease_abbr}？",
            f" {disease_abbr} 需要做什麼檢查嗎？",
            f"我得了 {disease_abbr}，去看醫生會有幫助嗎？"
        ]
    }

    if intent in question_templates:
        return random.choice(question_templates[intent])
    else:
        print(f"警告：未知意圖 '{intent}'，為疾病 '{disease_abbr}' 生成通用問題。")
        return f"關於 {disease_abbr} 的問題"

def create_natural_qa_from_csv(csv_file, output_csv_file):
    """
    從原始 CSV 創建自然問答資料集
    假設 CSV 格式為：input,target
    其中 input 是 "disease=BCC intent=treatment" 格式
    """
    df = pd.read_csv(csv_file)

    print("原始 CSV 欄位：", df.columns.tolist())
    print("前5行資料：")
    print(df.head())

    if 'input' not in df.columns or 'target' not in df.columns:
        print("請確認 CSV 檔案有 'input' 和 'target' 欄位")
        return

    expanded_data = []

    for _, row in df.iterrows():
        structured_input = str(row['input']) # 例如 "disease=BCC intent=treatment"
        response = str(row['target'])

        disease_abbr, intent = parse_structured_input(structured_input)

        # 取得疾病的中文名稱，如果字典中沒有則使用縮寫
        disease_chinese_name = disease_chinese_name_map.get(disease_abbr, disease_abbr)

        # 為每個回答生成多個自然問題，並將結構化資訊作為前綴
        for _ in range(5):  # 每個回答生成5個不同的問題，增加訓練樣本數量
            # 使用疾病縮寫來生成自然問題，確保問題模板中的 {disease} 得到正確替換
            natural_question = generate_natural_questions(disease_abbr, intent)

            # *** 關鍵修改：將 formatted_question 的格式調整為與 app.py 中的 contextual_input 一致 ***
            # 格式："{中文疾病名稱} (疾病縮寫) 相關問題：{用戶問題}"
            # 在訓練數據生成時，我們假設 last_pred_disease 就是当前的 disease_abbr
            formatted_question = f"{disease_chinese_name} ({disease_abbr}) 相關問題：{natural_question}"

            expanded_data.append({
                'input': formatted_question, # 使用新的統一格式
                'target': response,
                'original_input': structured_input,
                'disease': disease_abbr, # 這裡保持疾病縮寫，便於查看
                'intent': intent
            })

    expanded_df = pd.DataFrame(expanded_data)
    expanded_df.to_csv(output_csv_file, index=False, encoding='utf-8')

    print(f"\n擴充資料集已儲存至 {output_csv_file}")
    print(f"原始資料: {len(df)} 條")
    print(f"擴充後: {len(expanded_data)} 條問答對")

    print("\n擴充後的資料範例：")
    for i in range(min(5, len(expanded_data))):
        print(f"問題: {expanded_data[i]['input']}")
        print(f"回答: {expanded_data[i]['target']}")
        print(f"原始結構化輸入: {expanded_data[i]['original_input']}")
        print(f"解析疾病: {expanded_data[i]['disease']}")
        print(f"解析意圖: {expanded_data[i]['intent']}\n")

    return expanded_data

# ---

# ===== 第二步：T5 模型訓練 =====

def train_medical_chatbot(original_csv_file):
    """
    訓練醫療聊天機器人
    """
    # 先擴充資料集
    print("正在從 CSV 創建自然問答資料集...")
    natural_qa_file = "natural_medical_qa.csv"
    create_natural_qa_from_csv(original_csv_file, natural_qa_file)

    # 載入擴充後的資料集
    df = pd.read_csv(natural_qa_file)
    # 只使用 input 和 target 欄位
    df_train = df[['input', 'target']].copy()

    dataset = Dataset.from_pandas(df_train)
    dataset = dataset.train_test_split(test_size=0.1)

    print(f"訓練集大小: {len(dataset['train'])}")
    print(f"測試集大小: {len(dataset['test'])}")

    # 載入 tokenizer 和模型
    try:
        # 正確導入和使用 MT5Tokenizer 和 MT5ForConditionalGeneration
        tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
        model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

        # 確保模型在 GPU 上
        if torch.cuda.is_available():
            model.to("cuda")
            print("模型已移至 CUDA 設備。")
        else:
            print("CUDA 不可用，模型將在 CPU 上運行。")

    except ImportError as e:
        print(f"T5 模型需要 SentencePiece 庫: {e}")
        print("請執行: pip install sentencepiece")
        print("或者我們可以改用 BART 模型...")
        # 如果 MT5 載入失敗，才考慮 BART
        from transformers import BartTokenizer, BartForConditionalGeneration
        print("改用 BART 模型...")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    # 定義最大輸入和目標長度
    max_input_length = 128
    max_target_length = 256

     # Tokenize 資料
    print("正在 tokenize 資料...")
    tokenized_dataset = dataset.map(
        lambda examples: {
            **tokenizer(
                [str(q) for q in examples["input"]], # 這裡不需要額外的 "醫療問答: " 前綴了，因為 formatted_question 已經包含了上下文
                truncation=True,
                max_length=max_input_length,
                return_tensors=None
            ),
            "labels": tokenizer(
                examples["target"],
                truncation=True,
                max_length=max_target_length,
                return_tensors=None
            )["input_ids"]
        },
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    print("Tokenization 完成！")

    # 創建數據整理器 (Data Collator)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # 訓練參數
    training_args = TrainingArguments(
        output_dir="./medical_chatbot_model",
        per_device_train_batch_size=8, # 通常 GPU 可以用更大的 batch size
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        learning_rate=2e-4,
        warmup_steps=500,
        save_strategy="epoch",
        evaluation_strategy="epoch",  # <--- 修正這裡：將 eval_strategy 改為 evaluation_strategy
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        no_cuda=True,  # <--- 如果有 GPU，請務必改為 False，以便使用 GPU 訓練
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 執行訓練
    print("開始訓練醫療聊天機器人...")
    trainer.train()

    # 儲存模型
    trainer.save_model("./medical_chatbot_final")
    tokenizer.save_pretrained("./medical_chatbot_final")

    print("訓練完成！模型已儲存至 ./medical_chatbot_final")

# ---

# ===== 第三步：聊天機器人推理 =====

class MedicalChatbot:
    def __init__(self, model_path="./medical_chatbot_final"):
        """
        初始化醫療聊天機器人
        """
        print("載入醫療聊天機器人...")
        try:
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

    def chat(self, question_with_disease, predicted_disease_abbr=None): # 新增 predicted_disease_abbr 參數
        """
        與聊天機器人對話。
        question_with_disease: 用戶輸入的純問題字串，例如 "這是什麼？"
        predicted_disease_abbr: 從圖像識別等模組得到的疾病縮寫，例如 "MEL"
        """
        if predicted_disease_abbr:
            disease_chinese_name = disease_chinese_name_map.get(predicted_disease_abbr, predicted_disease_abbr)
            # 構建與訓練時一致的輸入格式："{中文疾病名稱} (縮寫) 相關問題：{用戶問題}"
            input_text = f"{disease_chinese_name} ({predicted_disease_abbr}) 相關問題：{question_with_disease}"
        else:
            # 如果沒有疾病縮寫，則只使用用戶問題 (這通常不理想，因為模型可能無法理解上下文)
            # 訓練時的輸入格式是帶有疾病上下文的，因此這裡也應盡量提供
            input_text = question_with_disease
            print("警告: chat 函數沒有提供 predicted_disease_abbr。模型可能會因為缺少疾病上下文而表現不佳。")

        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=256,
                min_length=10,
                num_beams=6,  # 增加 beam_size 可能會獲得更連貫的回答
                temperature=0.8, # 溫度
                do_sample=True, # 啟用採樣
                top_k=50,       # top-k 採樣
                top_p=0.95,     # nucleus 採樣
                repetition_penalty=1.5, # 重複懲罰
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id # 確保有 eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._clean_response(response) # 使用清理函數

    def _clean_response(self, response):
        # 移除可能的特殊 token，例如 <pad>
        return response.replace("<pad>", "").strip()

    def batch_test(self, test_questions_with_disease): # 傳入包含疾病的測試問題列表
        """
        批量測試問題
        test_questions_with_disease 應該是 (user_input, disease_abbr) 的元組列表
        例如: [("BCC 要怎麼治療？", "BCC"), ("什麼是 AKIEC？", "AKIEC")]
        """
        print("\n=== 批量測試 ===")
        for i, (question, disease_abbr) in enumerate(test_questions_with_disease, 1):
            try:
                response = self.chat(question, predicted_disease_abbr=disease_abbr) # 傳入疾病縮寫
                print(f"{i}. 問: ({disease_abbr}) {question}")
                print(f"   答: {response}\n")
            except Exception as e:
                print(f"{i}. 問: ({disease_abbr}) {question}")
                print(f"   錯誤: {e}\n")

    def start_conversation(self):
        """
        開始對話模式
        """
        print("\n=== 醫療聊天機器人 ===")
        print("您好！我是醫療諮詢助手，有什麼皮膚相關問題嗎？")
        print("請先輸入您關心的疾病縮寫（例如：BCC, MEL 等），然後輸入問題。")
        print("(輸入 'quit' 結束對話)")

        current_disease_abbr = None

        while True:
            user_input = input("\n您 (疾病縮寫或問題): ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', '結束']:
                print("機器人: 謝謝使用，祝您健康！")
                break

            # 嘗試解析疾病縮寫
            # 檢查輸入是否是已知的疾病縮寫 (不區分大小寫)
            if user_input.upper() in disease_chinese_name_map:
                current_disease_abbr = user_input.upper()
                print(f"機器人: 好的，您目前關心的是 {disease_chinese_name_map[current_disease_abbr]} ({current_disease_abbr})。現在可以提問了。")
                continue

            if not user_input:
                continue

            if current_disease_abbr is None:
                print("機器人: 請先輸入您關心的疾病縮寫，例如 'BCC'，再提問。")
                continue

            try:
                # 傳入用戶問題和當前疾病縮寫
                response = self.chat(user_input, predicted_disease_abbr=current_disease_abbr)
                print(f"機器人: {response}")
            except Exception as e:
                print(f"機器人: 抱歉，我遇到了一些問題：{e}")

# ---

# ===== 主程式 =====

def main():
    """
    主要執行流程
    """
    # 你的原始 CSV 檔案名稱
    original_csv_file = "t5_intent_disease_qa_dataset_large.csv" # 確保這個檔案存在

    print("=== 醫療聊天機器人訓練系統 ===")
    print("1. 正在檢查資料集...")

    # 檢查檔案是否存在
    try:
        df = pd.read_csv(original_csv_file)
        print(f"找到資料集: {len(df)} 條資料")
        print("欄位:", df.columns.tolist())
    except FileNotFoundError:
        print(f"找不到檔案: {original_csv_file}")
        print("請確認檔案名稱和路徑正確")
        return

    # 詢問是否要重新訓練
    choice = input("\n要重新訓練模型嗎？(y/n): ").lower()

    if choice == 'y':
        # 訓練模型
        train_medical_chatbot(original_csv_file)

    # 測試聊天機器人
    try:
        chatbot = MedicalChatbot()

        # 預設測試問題，現在需要包含疾病縮寫 (與 app.py 傳入方式一致)
        test_questions_with_disease = [
            ("BCC 要怎麼治療？", "BCC"),
            ("BKL 會傳染嗎？", "BKL"),
            ("什麼是 AKIEC？", "AKIEC"),
            ("DF 危險嗎？", "DF"),
            ("NV 需要看醫生嗎？", "NV"),
            ("MEL 的治療方式有哪些？", "MEL")
        ]

        # 批量測試
        chatbot.batch_test(test_questions_with_disease)

        # 開始互動對話
        start_chat = input("要開始對話模式嗎？(y/n): ").lower()
        if start_chat == 'y':
            chatbot.start_conversation()

    except Exception as e:
        print(f"載入聊天機器人失敗: {e}")
        print("請先訓練模型 (或檢查 medical_chatbot_final 資料夾是否存在)")

if __name__ == "__main__":
    main()
