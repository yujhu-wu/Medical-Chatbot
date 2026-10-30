from flask import Flask, request, jsonify, render_template, send_file
import torch
from PIL import Image
import numpy as np
import io
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import textwrap
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The tokenizer class you load from this checkpoint is not the same type as the class this function is called from.*")
warnings.filterwarnings("ignore", category=UserWarning, message="You are using a model of type t5 to instantiate a model of type mt5. This is not supported for all configurations of models and can yield errors.*")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.*")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future.*")

# 從新的模組導入 MedicalChatbot 類別
from medical_chatbot_module import MedicalChatbot

# === 模型設定 ===

# FAT_Net 與其子類別定義 (保持不變)
from torch import nn
from torchvision import models as resnet_model

class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1
        return out

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return torch.mul(x, y)

class FAT_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(FAT_Net, self).__init__()
        # 這裡的 deit_tiny_distilled_patch16_224 是 FAT_Net 內部使用的 Transformer
        transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList([transformer.blocks[i] for i in range(12)])
        self.conv_seq_img = nn.Conv2d(192, 512, kernel_size=1)
        self.se = SEBlock(1024)
        self.conv2d = nn.Conv2d(1024, 512, kernel_size=1)
        self.FAMBlock1 = FAMBlock(64)
        self.FAMBlock2 = FAMBlock(128)
        self.FAMBlock3 = FAMBlock(256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for _ in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for _ in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for _ in range(2)])
        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])
        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)
        emb = self.patch_embed(x)
        for i in range(12):
            emb = self.transformers[i](emb)
        feature_tf = emb.permute(0, 2, 1).view(b, 192, 14, 14)
        feature_tf = self.conv_seq_img(feature_tf)
        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)
        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        return out

SIZE = 224 # 保留 FAT_Net 的 SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 疾病分類標籤 (請確保與你的訓練數據和模型輸出類別數匹配)
# 如果你的分類模型有 10 個輸出類別，這裡也應該有 10 個對應
label_2_id = {
    'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6,
    'UNK7': 7, 'UNK8': 8, 'UNK9': 9 # 這些是假設的額外類別，請根據你的實際數據替換
}
id_2_label = {v: k for k, v in label_2_id.items()}

# 疾病文字說明 (用於顯示給用戶和傳給聊天機器人)
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

import timm
import os

# === 載入分類模型 (Swin Transformer) ===
model_name_classify = 'swin_base_patch4_window12_384'
# 請確認你的模型路徑是否正確，這裡使用的是硬編碼路徑
checkpoint_path_classify = "skin_model/resnext50_32x4d_fold2_best_epoch8.pth"

print(f"嘗試載入分類模型: {model_name_classify}...")

# 根據先前的錯誤訊息，模型權重是為 10 個類別訓練的
model_classify = timm.create_model(model_name_classify, pretrained=False, num_classes=len(label_2_id)) # 修正為 label_2_id 的長度

if not os.path.exists(checkpoint_path_classify):
    raise FileNotFoundError(f"分類模型 checkpoint 檔案未找到: {checkpoint_path_classify}")

# 載入整個 checkpoint
model_data_classify = torch.load(
    checkpoint_path_classify,
    map_location=DEVICE
)
# 檢查載入的 checkpoint 是否是一個字典，並且包含 'model' 鍵
if isinstance(model_data_classify, dict) and 'model' in model_data_classify:
    checkpoint_state_dict_classify = model_data_classify['model']
    print("Checkpoint 載入為字典，並提取 'model' 鍵下的狀態字典。")
else:
    # 如果不是字典或沒有 'model' 鍵，則假設它直接是狀態字典
    checkpoint_state_dict_classify = model_data_classify
    print("Checkpoint 直接載入為狀態字典。")

# 嘗試載入 state_dict (現在應該能正常載入)
try:
    model_classify.load_state_dict(checkpoint_state_dict_classify)
    print(f"\n{model_name_classify} 模型成功載入！")
except RuntimeError as e:
    print(f"\n載入 {model_name_classify} 權重時發生錯誤: {e}")
    print("這表示權重檔案的鍵名與模型結構可能不完全匹配，或 head 層類別數不符。")
    print("嘗試使用 strict=False 載入部分匹配的權重，並檢查 head 層。")

    missing_keys, unexpected_keys = model_classify.load_state_dict(checkpoint_state_dict_classify, strict=False)
    if missing_keys:
        print(f"\n模型中缺少但 checkpoint 中沒有的鍵 (將使用模型預設初始化): {missing_keys}")
    if unexpected_keys:
        print(f"Checkpoint 中有但模型中不需要的鍵 (已忽略): {unexpected_keys}")
    print(f"\n{model_name_classify} 模型部分載入完成 (使用 strict=False)。")

model_classify.to(DEVICE)
model_classify.eval()


# === 載入分割模型 ===
# FAT_Net 類別已經在前面定義，所以直接實例化
model_segment = FAT_Net().to(DEVICE)
seg_path = "skin_model/checkpoint1017_last.pth.tar" # 請確認你的模型路徑是否正確
if not os.path.exists(seg_path):
    raise FileNotFoundError(f"分割模型 checkpoint 檔案未找到: {seg_path}")
# 注意：這裡的權重檔結構應該是一個字典，其中包含 'state_dict' 鍵
# 1. 先 load 完整的 checkpoint（不要傳 weights_only）
checkpoint = torch.load(seg_path, map_location=DEVICE)
# 2. 把真正的 state_dict 給模型
#    假設你的檔案裡 state_dict 是存放在 'state_dict' 這個 key：
model_segment.load_state_dict(checkpoint['state_dict'])
model_segment.eval()
print("分割模型 FAT_Net 載入完成！")


# === 載入醫療聊天機器人模型 ===
try:
    # 這裡直接實例化 MedicalChatbot，其內部會處理載入邏輯
    # 模型的路徑應該與 train_medical_chatbot.py 中保存的路徑一致
    medical_chatbot_model = MedicalChatbot(model_path="./medical_chatbot_final")
    # 這裡的 print 已經在 MedicalChatbot 類的 __init__ 裡
except Exception as e:
    print(f"載入醫療聊天機器人模型失敗: {e}")
    # 如果載入失敗，可以考慮設定一個 None 或預設行為
    medical_chatbot_model = None


# 分類模型圖片前處理：Swin Transformer 通常使用 384x384 輸入
transform_classify_img = transforms.Compose([
    transforms.Resize((384, 384)), # Swin Transformer 通常使用 384 輸入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
])


# === App 啟動 ===
app = Flask(__name__)
last_pred_disease = "" # 初始化為空字串，或 'Unknown'

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_pred_disease
    # 當頁面刷新或首次載入時，重置疾病狀態
    last_pred_disease = ""
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    global last_pred_disease
    file = request.files.get('file')
    if not file:
        return jsonify({"predicted": "Unknown"})

    try:
        img_pil = Image.open(file).convert('RGB')

        # 使用分類模型的前處理
        img_tensor = transform_classify_img(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model_classify(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_id = torch.argmax(probs, dim=1).item()

            # 確保 pred_id 在 id_2_label 的範圍內，以避免 KeyError
            if pred_id in id_2_label:
                pred_label = id_2_label.get(pred_id, "Unknown")
            else:
                pred_label = "Unknown" # 或者您可以將其映射到一個通用類別，例如 'OTHER'

            last_pred_disease = pred_label if pred_label in label_2_id else "Unknown"

        return jsonify({
            "predicted": last_pred_disease,
            "disease_name": info_dict.get(last_pred_disease, "未知疾病") # 未知疾病也給予預設說明
        })

    except Exception as e:
        print(f"圖片上傳處理錯誤: {e}")
        return jsonify({"predicted": "Unknown", "disease_name": "處理圖片時發生錯誤"})


@app.route('/chat', methods=['POST'])
def chat():
    global last_pred_disease
    data = request.get_json()
    user_input = data.get("message", "")

    if last_pred_disease == "Unknown" or last_pred_disease == "":
        return jsonify({"reply": "目前尚未取得正確疾病資訊，請先上傳圖片進行診斷。"})

    if medical_chatbot_model is None:
        return jsonify({"reply": "聊天機器人模型未成功載入。"})

    # 取得疾病中文名稱
    # 這裡的 last_pred_disease 是疾病的縮寫 (如 "BCC")
    disease_chinese_name = info_dict.get(last_pred_disease, last_pred_disease)

    # 構建與訓練時完全一致的輸入格式
    # 格式："{disease_chinese_name} ({disease_abbr}) 相關問題：{user_input}"
    contextual_input = f"{disease_chinese_name} ({last_pred_disease}) 相關問題：{user_input}"

    print(f"傳給聊天機器人的輸入: {contextual_input}") # 除錯用

    # 使用醫療聊天機器人模型進行對話
    response = medical_chatbot_model.chat(contextual_input)

    return jsonify({"reply": response})

if __name__ == '__main__':
    app.run(debug=True)
