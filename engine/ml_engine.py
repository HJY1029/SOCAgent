import os
import numpy as np
import pandas as pd
import joblib
from dpkt import pcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import rdpcap
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from idstools import rule
import json
import time
import warnings
import glob
from urllib.request import urlretrieve
from decimal import Decimal
import math
from collections import defaultdict
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

warnings.filterwarnings('ignore')

# 配置参数
MODELS_DIR = "./ml_models"
DATASETS_DIR = "./datasets"
PCAP_FILE = "sample.pcap"
OUTPUT_FILE = "alerts_with_ml.json"
MODEL_TYPES = ["isolation_forest", "xgboost", "lightgbm", "random_forest"]
# CICIDS2017 全量约 280 万行，易触发 OOM；限制最大样本数避免被系统 kill
CICIDS2017_MAX_SAMPLES = int(os.environ.get("CICIDS2017_MAX_SAMPLES", "250000"))

# KDD99和CICIDS2017数据集特征映射
KDD99_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

CICIDS2017_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
    'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

ATTACK_TYPE_MAPPING = {
    "kdd99": {
        "normal": "正常流量",
        "back": "后门攻击",
        "buffer_overflow": "缓冲区溢出攻击",
        "ftp_write": "FTP写攻击",
        "guess_passwd": "密码猜测攻击",
        "imap": "IMAP攻击",
        "ipsweep": "IP扫描",
        "land": "LAND攻击",
        "loadmodule": "模块加载攻击",
        "multihop": "多跳攻击",
        "neptune": "SYN洪水攻击",
        "nmap": "端口扫描",
        "perl": "Perl攻击",
        "phf": "PHF攻击",
        "pod": "POD攻击",
        "portsweep": "端口扫描",
        "rootkit": "Rootkit攻击",
        "satan": "Satan扫描",
        "smurf": "Smurf攻击",
        "spy": "间谍软件",
        "teardrop": "泪滴攻击",
        "warezclient": "软件盗版客户端",
        "warezmaster": "软件盗版主机"
    },
    "cicids2017": {
        "BENIGN": "正常流量",
        "DDoS": "分布式拒绝服务攻击",
        "PortScan": "端口扫描",
        "Bot": "僵尸网络活动",
        "Infiltration": "渗透攻击",
        "Web Attack": "Web应用攻击",
        "Brute Force": "暴力破解",
        "SQL Injection": "SQL注入",
        "FTP-Patator": "FTP暴力破解",
        "SSH-Patator": "SSH暴力破解",
        "DoS": "拒绝服务攻击",
        # 添加二分类标签
        "normal": "正常流量",
        "attack": "恶意流量"
    }
}



# ==================== 机器学习模型引擎 ====================
class MLEngine:
    def __init__(self, model_type, dataset_type="kdd99"):
        self.original_categorical_features = None
        self.original_features = None
        self.initialized = False
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features = KDD99_FEATURES if dataset_type == "kdd99" else CICIDS2017_FEATURES
        if dataset_type == "kdd99":
            self.categorical_features = ['protocol_type', 'service', 'flag']
        else:  # CICIDS2017
            self.categorical_features = []  # CICIDS2017没有分类特征

        self.required_features = []
        self.ATTACK_TYPE_MAPPING = ATTACK_TYPE_MAPPING

        # 确保模型目录存在
        os.makedirs(MODELS_DIR, exist_ok=True)

        # 模型文件路径
        model_prefix = f"{model_type}_{dataset_type}"
        self.model_path = os.path.join(MODELS_DIR, f"{model_prefix}_model.pkl")
        self.scaler_path = os.path.join(MODELS_DIR, f"{model_prefix}_scaler.pkl")
        self.encoder_path = os.path.join(MODELS_DIR, f"{model_prefix}_encoder.pkl")
        self.feature_path = os.path.join(MODELS_DIR, f"{model_prefix}_features.pkl")

        # 尝试加载现有模型
        if (os.path.exists(self.model_path) and
                os.path.exists(self.scaler_path) and
                os.path.exists(self.feature_path)):
            self.load_model()
            self.initialized = True
        else:
            print(f"未找到预训练模型，开始训练新的{model_type.upper()}模型 ({dataset_type})...")
            self.train_model()
            self.initialized = True

    def is_initialized(self):
        return self.initialized

    def load_model(self):
        """加载预训练模型和预处理工具"""
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.required_features = joblib.load(self.feature_path)

        if os.path.exists(self.encoder_path):
            self.label_encoder = joblib.load(self.encoder_path)
        print(f"已加载{self.model_type.upper()}模型 ({self.dataset_type})")


    def _safe_label_encode(self, labels):
        """安全处理标签编码，处理单类别情况"""
        unique_labels = np.unique(labels)

        # 如果只有一个类别，添加虚拟类别
        if len(unique_labels) == 1:
            print(f"警告：标签中只有一个类别 '{unique_labels[0]}'，添加虚拟类别")
            # 创建带虚拟类别的编码器
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit([unique_labels[0], "dummy_class"])

            # 转换标签，所有实际标签映射到0，虚拟标签映射到1
            encoded = np.zeros(len(labels), dtype=int)
            return encoded, True
        else:
            # 多个类别，正常处理
            self.label_encoder = LabelEncoder()
            encoded = self.label_encoder.fit_transform(labels)
            return encoded, False

    def load_dataset(self):
        """加载网络安全数据集"""
        if self.dataset_type == "kdd99":
            # 从备用源下载KDD99数据集
            dataset_path = os.path.join(DATASETS_DIR, "kddcup.data_10_percent.gz")
            if not os.path.exists(dataset_path):
                print("下载KDD99数据集...")
                url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
                os.makedirs(DATASETS_DIR, exist_ok=True)
                urlretrieve(url, dataset_path)

            df = pd.read_csv(dataset_path, compression='gzip', header=None,
                             names=KDD99_FEATURES + ['label'])

            # +++ 修复：统一正常样本标签 +++
            df['label'] = df['label'].replace('normal.', 'normal')

            # +++ 修复：移除添加模拟样本的逻辑 +++
            print("KDD99数据集加载完成，正常样本数量:", len(df[df['label'] == 'normal']))

        else:  # CICIDS2017
            # 需要提前下载数据集并解压
            dataset_path = os.path.join(DATASETS_DIR, "MachineLearningCVE", "*.csv")
            files = glob.glob(dataset_path)
            if not files:
                raise FileNotFoundError("CICIDS2017数据集未找到，请下载并解压到datasets/MachineLearningCVE/")

            df_list = []
            for file in files:
                try:
                    df_chunk = pd.read_csv(file, low_memory=False)
                    df_list.append(df_chunk)
                except Exception as e:
                    print(f"加载文件{file}失败: {str(e)}")
            df = pd.concat(df_list, ignore_index=True)

            # 重命名列以匹配标准特征名（strip 并处理重复列名，与 CICIDS2017_FEATURES 一致）
            raw_cols = [c.strip() for c in df.columns]
            # 处理 CSV 中可能存在的重复列名（如两个 "Fwd Header Length"）→ 第二个改为 "Fwd Header Length.1"
            seen = {}
            new_columns = []
            for c in raw_cols:
                if c == 'Label':
                    new_columns.append('label')
                    continue
                if c in seen:
                    seen[c] += 1
                    new_columns.append(f"{c}.1" if seen[c] == 2 else f"{c}.{seen[c]}")
                else:
                    seen[c] = 1
                    new_columns.append(c)
            df.columns = new_columns

            # 修复CICIDS2017标签名称不一致的问题
            df['label'] = df['label'].str.strip()
            df['label'] = df['label'].replace({
                'Web Attack � Brute Force': 'Web Attack',
                'Web Attack � XSS': 'Web Attack',
                'Web Attack � Sql Injection': 'Web Attack',
                'DoS Hulk': 'DoS',
                'DoS GoldenEye': 'DoS',
                'DoS slowloris': 'DoS',
                'DoS Slowhttptest': 'DoS',
                'FTP-Patator': 'FTP-Patator',
                'SSH-Patator': 'SSH-Patator'
            })

            # +++ 修复：处理Heartbleed和Infiltration样本过少的问题 +++
            # 对样本量过少的类别进行过采样
            min_samples = 1000  # 每个类别至少1000个样本
            label_counts = df['label'].value_counts()
            for label, count in label_counts.items():
                if count < min_samples and label != 'BENIGN':
                    print(f"对{label}进行过采样 (当前样本数: {count})")
                    label_df = df[df['label'] == label]
                    oversampled = label_df.sample(min_samples - count, replace=True, random_state=42)
                    df = pd.concat([df, oversampled], ignore_index=True)

            # +++ 限制总样本数，避免 OOM 被系统 Kill +++
            if len(df) > CICIDS2017_MAX_SAMPLES:
                print(f"数据集共 {len(df)} 行，为控制内存进行分层采样至 {CICIDS2017_MAX_SAMPLES} 行（可设环境变量 CICIDS2017_MAX_SAMPLES 调整）")
                frac = CICIDS2017_MAX_SAMPLES / len(df)
                df = df.groupby('label', group_keys=False).apply(
                    lambda g: g.sample(n=min(len(g), max(1, int(round(frac * len(g))))), random_state=42)
                ).reset_index(drop=True)
                if len(df) > CICIDS2017_MAX_SAMPLES:
                    df = df.sample(n=CICIDS2017_MAX_SAMPLES, random_state=42).reset_index(drop=True)
                print(f"采样后: {len(df)} 行")

        return df

    def preprocess_data(self, df):
        """数据预处理"""
        # 选择特征
        df = df[self.features + ['label']]

        # 处理分类特征
        if self.dataset_type == "kdd99":
            # 创建特征副本用于后续匹配
            original_features = self.features.copy()

            # 执行one-hot编码
            df = pd.get_dummies(df, columns=self.categorical_features)

            # 更新特征列表
            self.features = [col for col in df.columns if col != 'label']

            # 保存原始特征名用于预测时匹配
            self.original_categorical_features = self.categorical_features
            self.original_features = original_features
        else:  # CICIDS2017
            # 移除包含无穷大或NaN的列
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            # 确保特征顺序一致
            available_features = [f for f in self.features if f in df.columns]
            df = df[available_features + ['label']]
            self.features = available_features

        # 处理缺失值
        df.fillna(0, inplace=True)

        # +++ 新增：处理无穷大值 +++
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # +++ 修复：对于Isolation Forest模型，将CICIDS2017转换为二分类问题 +++
        if self.model_type == "isolation_forest" and self.dataset_type == "cicids2017":
            # 将标签转换为二分类：正常流量和攻击
            df['label'] = df['label'].apply(lambda x: 'normal' if x == 'BENIGN' else 'attack')
            print("已将CICIDS2017转换为二分类问题（正常流量 vs 攻击）")

        # 简化标签（二分类：正常/异常）
        if self.dataset_type == "kdd99":
            df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
        elif self.dataset_type == "cicids2017" and self.model_type != "isolation_forest":
            # 保留多类别标签
            df['label'] = df['label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else x)

        # +++ 修复：样本平衡处理 +++
        if self.dataset_type == "cicids2017" and self.model_type != "isolation_forest":
            # 对正常样本进行欠采样，防止占比过高
            normal_samples = df[df['label'] == 'BENIGN']
            attack_samples = df[df['label'] != 'BENIGN']

            # 控制正常样本数量不超过攻击样本的10倍
            max_normal = min(len(normal_samples), len(attack_samples) * 10)
            if len(normal_samples) > max_normal:
                print(f"对正常样本进行欠采样: {len(normal_samples)} -> {max_normal}")
                normal_samples = normal_samples.sample(max_normal, random_state=42)
                df = pd.concat([normal_samples, attack_samples], ignore_index=True)

        # 分离特征和标签
        X = df.drop('label', axis=1)
        y = df['label']

        # 确保所有特征都是数值类型
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 修复：确保所有列都是数值类型
        for col in X.columns:
            # 处理空列
            if X[col].nunique() == 0:
                X[col] = 0.0
                continue

            # 转换非数值列
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])

            # 转换为float32
            X[col] = X[col].astype(np.float32).fillna(0)

        # +++ 新增：特征选择 +++
        if self.dataset_type == "cicids2017":
            # 移除低方差特征
            selector = VarianceThreshold(threshold=0.01)
            X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])

            # +++ 新增：基于相关性的特征选择 +++
            k = min(50, X.shape[1])  # 最多选择50个特征
            if k > 0:
                selector = SelectKBest(f_classif, k=k)
                X = pd.DataFrame(selector.fit_transform(X, y), columns=X.columns[selector.get_support()])

            print(f"特征选择后保留 {X.shape[1]} 个特征")

        return X, y

    def train_model(self):
        """训练机器学习模型"""
        try:
            # 加载数据集
            print(f"加载{self.dataset_type.upper()}数据集...")
            df = self.load_dataset()

            # +++ 打印数据分布 +++
            print("数据集分布:")
            print(df['label'].value_counts())

            # 数据预处理
            X, y = self.preprocess_data(df)

            # 检查标签多样性
            unique_labels = np.unique(y)
            print(f"标签类别: {unique_labels}")

            # 划分训练测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 保存特征列表
            self.required_features = X_train.columns.tolist()
            joblib.dump(self.required_features, self.feature_path)
            print(f"保存了 {len(self.required_features)} 个特征")

            # 特征缩放
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            print("特征缩放完成")

            # 创建和训练模型
            if self.model_type == "isolation_forest":
                print("训练Isolation Forest模型...")

                # +++ 修复：异常检测需要正常样本 +++
                normal_mask = (y_train == 'normal') | (y_train == 'BENIGN')
                normal_count = normal_mask.sum()

                if normal_count == 0:
                    print("错误：没有正常样本，无法训练异常检测模型")
                    return

                # 计算异常比例
                anomaly_ratio = 1.0 - (normal_count / len(y_train))
                anomaly_ratio = max(0.01, min(anomaly_ratio, 0.5))  # 限制在合理范围
                print(f"异常比例: {anomaly_ratio:.4f}")

                self.model = IsolationForest(
                    n_estimators=300,  # 增加树的数量
                    contamination=anomaly_ratio,
                    max_samples=min(256, normal_count),  # 限制样本量
                    random_state=42,
                    n_jobs=-1
                )

                # 使用正常样本训练模型
                self.model.fit(X_train[normal_mask])

                # 评估模型
                y_pred = self.model.predict(X_test)
                # 将异常分数转换为标签（1表示正常，-1表示异常）
                y_pred_labels = ["attack" if x == -1 else "normal" for x in y_pred]
                y_true = y_test

                # 对于二分类问题，打印ROC AUC
                if self.dataset_type == "cicids2017":
                    # 计算异常分数（离群程度）
                    decision_scores = self.model.decision_function(X_test)
                    # 将分数转换为0-1范围（0表示正常，1表示异常）
                    normalized_scores = 1 / (1 + np.exp(-decision_scores))
                    roc_auc = roc_auc_score(y_true == 'attack', normalized_scores)

            else:
                print(f"训练{self.model_type.upper()}分类模型...")

                # 修复：确保标签是一维数组
                if y_train.ndim > 1:
                    y_train = y_train.ravel()
                if y_test.ndim > 1:
                    y_test = y_test.ravel()

                # 安全处理标签编码
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(y_train)  # 使用训练集标签拟合

                # 检查标签类别
                if len(self.label_encoder.classes_) == 1:
                    # 单类别处理：添加虚拟类别
                    print(f"警告：数据中只有一个类别 '{self.label_encoder.classes_[0]}'，添加虚拟类别")
                    dummy_label = "dummy_class"
                    self.label_encoder.classes_ = np.append(self.label_encoder.classes_, dummy_label)

                    # 为训练集添加一个虚拟样本
                    dummy_sample = np.zeros(X_train.shape[1])
                    X_train = np.vstack([X_train, dummy_sample])
                    y_train = np.append(y_train, dummy_label)

                # 转换标签
                y_train_encoded = self.label_encoder.transform(y_train)
                y_test_encoded = self.label_encoder.transform(y_test)

                # +++ 处理类别不平衡 +++
                from sklearn.utils.class_weight import compute_class_weight
                class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
                weight_dict = dict(enumerate(class_weights))

                # 创建模型
                if self.model_type == "xgboost":
                    self.model = XGBClassifier(
                        n_estimators=300,  # 增加树的数量
                        max_depth=8,  # 增加深度
                        learning_rate=0.05,  # 降低学习率
                        use_label_encoder=False,
                        eval_metric='logloss',
                        scale_pos_weight=weight_dict[1] if len(weight_dict) == 2 else None,
                        subsample=0.8,  # 防止过拟合
                        colsample_bytree=0.8,
                        n_jobs=-1
                    )
                elif self.model_type == "lightgbm":
                    self.model = LGBMClassifier(
                        n_estimators=300,
                        max_depth=8,
                        learning_rate=0.05,
                        class_weight='balanced',  # 处理不平衡
                        n_jobs=-1,
                        verbosity=-1,  # 关闭训练日志，避免刷屏
                        force_col_wise=True,  # 去掉 "Auto-choosing col-wise" 提示
                        min_child_samples=30,  # 叶节点最少样本数，减少 "No further splits" 警告
                    )
                elif self.model_type == "random_forest":
                    self.model = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        random_state=42,
                        class_weight='balanced',  # 处理不平衡
                        n_jobs=-1
                    )

                # 训练模型
                self.model.fit(X_train, y_train_encoded)

                # 评估模型
                y_pred = self.model.predict(X_test)
                y_pred_labels = self.label_encoder.inverse_transform(y_pred)
                y_true_labels = self.label_encoder.inverse_transform(y_test_encoded)

                # 过滤掉虚拟类别
                if "dummy_class" in y_pred_labels:
                    # 创建掩码过滤虚拟类别
                    mask = y_pred_labels != "dummy_class"
                    y_pred_labels = y_pred_labels[mask]
                    y_true_labels = y_true_labels[mask]
                    print(f"过滤掉 {np.sum(~mask)} 个虚拟样本")

                # 打印分类报告
                print(f"\n{self.model_type.upper()}模型评估报告 ({self.dataset_type}):")
                print(classification_report(y_true_labels, y_pred_labels))

                # +++ 对于多分类问题，打印混淆矩阵 +++
                if self.dataset_type == "cicids2017":
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=self.label_encoder.classes_)
                    print("\n混淆矩阵:")
                    print(pd.DataFrame(cm, index=self.label_encoder.classes_, columns=self.label_encoder.classes_))

            # 保存模型
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            if self.label_encoder:
                joblib.dump(self.label_encoder, self.encoder_path)
            print(f"模型已保存到 {self.model_path}")

            # 标记为已初始化
            self.initialized = True

        except Exception as e:
            print(f"训练失败: {str(e)}")
            import traceback
            traceback.print_exc()

            # 删除无效模型文件
            for path in [self.model_path, self.scaler_path, self.encoder_path, self.feature_path]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"已删除无效文件: {path}")

            # 重新抛出异常以便上层处理
            raise RuntimeError(f"模型训练失败: {str(e)}")

    def _extract_cicids2017_features(self, pcap_file):
        """为CICIDS2017数据集提取特征"""
        print(f"从 {pcap_file} 提取CICIDS2017特征...")
        packets = rdpcap(pcap_file)
        flows = {}
        metadata_list = []  # 存储流的元数据

        # 添加调试计数器
        packet_count = 0
        flow_count = 0

        # 遍历所有数据包
        for i, pkt in enumerate(packets):
            packet_count += 1

            # 检查是否有IP层
            if IP not in pkt:
                continue

            # 关键修复：转换时间戳为float
            packet_time = float(pkt.time)

            ip = pkt[IP]
            proto = ip.proto
            src = ip.src
            dst = ip.dst

            # 获取端口信息 - 转换为整数
            sport, dport = 0, 0
            if TCP in pkt:
                transport = pkt[TCP]
                sport = int(transport.sport)
                dport = int(transport.dport)
            elif UDP in pkt:
                transport = pkt[UDP]
                sport = int(transport.sport)
                dport = int(transport.dport)

            # 创建流标识 - 使用整数端口
            flow_id = (src, dst, sport, dport, proto)
            reverse_flow_id = (dst, src, dport, sport, proto)

            # 确定流方向
            if flow_id in flows:
                direction = "forward"
                actual_flow_id = flow_id
            elif reverse_flow_id in flows:
                direction = "backward"
                actual_flow_id = reverse_flow_id
            else:
                flow_count += 1
                direction = "forward"
                actual_flow_id = flow_id
                flows[actual_flow_id] = {
                    'src_ip': src,
                    'dst_ip': dst,
                    'src_port': sport,
                    'dst_port': dport,
                    'protocol': proto,
                    'start_time': packet_time,
                    'end_time': packet_time,
                    'fwd_packets': 0,
                    'bwd_packets': 0,
                    'fwd_bytes': 0,
                    'bwd_bytes': 0,
                    'fwd_packet_lengths': [],
                    'bwd_packet_lengths': [],
                    'fwd_iat': [],  # 前向包到达时间间隔
                    'bwd_iat': [],  # 后向包到达时间间隔
                    'fwd_header_lengths': [],  # 前向包头长度
                    'bwd_header_lengths': [],  # 后向包头长度
                    'flags': [],  # TCP标志
                    'last_fwd_time': None,  # 前向最后一个包时间
                    'last_bwd_time': None,  # 后向最后一个包时间
                    'active_times': [],  # 活动时间（包之间）
                    'idle_times': [],  # 空闲时间（流内）
                    'last_packet_time': packet_time  # 最后一个包时间
                }

            flow = flows[actual_flow_id]
            flow['end_time'] = packet_time
            flow['last_packet_time'] = packet_time

            # 更新活动时间和空闲时间
            if flow['last_packet_time'] is not None:
                idle_time = packet_time - flow['last_packet_time']
                flow['idle_times'].append(idle_time)
            flow['last_packet_time'] = packet_time

            # 计算包头长度
            header_length = len(ip) - len(ip.data) if hasattr(ip, 'data') else 0

            if direction == "forward":
                flow['fwd_packets'] += 1
                flow['fwd_bytes'] += len(pkt)
                flow['fwd_packet_lengths'].append(len(pkt))
                flow['fwd_header_lengths'].append(header_length)

                # 更新前向IAT
                if flow['last_fwd_time'] is not None:
                    iat = packet_time - flow['last_fwd_time']
                    flow['fwd_iat'].append(iat)
                    flow['active_times'].append(iat)
                flow['last_fwd_time'] = packet_time
            else:  # 反向流
                flow['bwd_packets'] += 1
                flow['bwd_bytes'] += len(pkt)
                flow['bwd_packet_lengths'].append(len(pkt))
                flow['bwd_header_lengths'].append(header_length)

                # 更新后向IAT
                if flow['last_bwd_time'] is not None:
                    iat = packet_time - flow['last_bwd_time']
                    flow['bwd_iat'].append(iat)
                    flow['active_times'].append(iat)
                flow['last_bwd_time'] = packet_time

            # 记录TCP标志
            if TCP in pkt:
                tcp = pkt[TCP]
                flags = ""
                if tcp.flags & 0x01: flags += "F"  # FIN
                if tcp.flags & 0x02: flags += "S"  # SYN
                if tcp.flags & 0x04: flags += "R"  # RST
                if tcp.flags & 0x08: flags += "P"  # PSH
                if tcp.flags & 0x10: flags += "A"  # ACK
                if tcp.flags & 0x20: flags += "U"  # URG
                if tcp.flags & 0x40: flags += "E"  # ECE
                if tcp.flags & 0x80: flags += "C"  # CWR
                flow['flags'].append(flags)

        print(f"处理完成: 总包数={packet_count}, 总流数={flow_count}")

        # 构建特征DataFrame
        features_list = []
        for flow_id, flow in flows.items():
            try:
                # 计算流持续时间（添加最小保护）
                duration = max(flow['end_time'] - flow['start_time'], 1e-6)  # 避免除0

                # 计算基本流特征
                total_fwd_packets = flow['fwd_packets']
                total_bwd_packets = flow['bwd_packets']
                total_fwd_bytes = flow['fwd_bytes']
                total_bwd_bytes = flow['bwd_bytes']

                # 计算包长度统计
                all_packet_lengths = flow['fwd_packet_lengths'] + flow['bwd_packet_lengths']

                # 安全统计函数
                def safe_stat(lst, default=0.0):
                    if not lst:
                        return default, default, default, default
                    arr = np.array(lst)
                    return (
                        float(np.max(arr)),
                        float(np.min(arr)),
                        float(np.mean(arr)),
                        float(np.std(arr))
                    )

                # 计算各种统计量
                fwd_pl_max, fwd_pl_min, fwd_pl_mean, fwd_pl_std = safe_stat(flow['fwd_packet_lengths'])
                bwd_pl_max, bwd_pl_min, bwd_pl_mean, bwd_pl_std = safe_stat(flow['bwd_packet_lengths'])
                all_pl_max, all_pl_min, all_pl_mean, all_pl_std = safe_stat(all_packet_lengths)

                # 计算IAT统计
                flow_iat = flow['fwd_iat'] + flow['bwd_iat']
                flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = safe_stat(flow_iat)
                fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = safe_stat(flow['fwd_iat'])
                bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = safe_stat(flow['bwd_iat'])

                # 计算流量速率
                flow_bytes_per_sec = (total_fwd_bytes + total_bwd_bytes) / max(duration, 1e-6)
                flow_packets_per_sec = (total_fwd_packets + total_bwd_packets) / max(duration, 1e-6)
                fwd_packets_per_sec = total_fwd_packets / max(duration, 1e-6)
                bwd_packets_per_sec = total_bwd_packets / max(duration, 1e-6)

                # 计算TCP标志计数
                flag_counts = defaultdict(int)
                for flags in flow['flags']:
                    for flag in flags:
                        flag_counts[flag] += 1

                # 计算包头统计
                fwd_header_mean, _, _, fwd_header_std = safe_stat(flow['fwd_header_lengths'])
                bwd_header_mean, _, _, bwd_header_std = safe_stat(flow['bwd_header_lengths'])

                # 计算活动时间和空闲时间统计
                active_mean, active_std, active_max, active_min = safe_stat(flow['active_times'])
                idle_mean, idle_std, idle_max, idle_min = safe_stat(flow['idle_times'])

                # 构建特征字典
                feature = {
                    'Flow Duration': duration,
                    'Total Fwd Packets': total_fwd_packets,
                    'Total Backward Packets': total_bwd_packets,
                    'Total Length of Fwd Packets': total_fwd_bytes,
                    'Total Length of Bwd Packets': total_bwd_bytes,
                    'Fwd Packet Length Max': fwd_pl_max,
                    'Fwd Packet Length Min': fwd_pl_min,
                    'Fwd Packet Length Mean': fwd_pl_mean,
                    'Fwd Packet Length Std': fwd_pl_std,
                    'Bwd Packet Length Max': bwd_pl_max,
                    'Bwd Packet Length Min': bwd_pl_min,
                    'Bwd Packet Length Mean': bwd_pl_mean,
                    'Bwd Packet Length Std': bwd_pl_std,
                    'Flow Bytes/s': flow_bytes_per_sec,
                    'Flow Packets/s': flow_packets_per_sec,
                    'Flow IAT Mean': flow_iat_mean,
                    'Flow IAT Std': flow_iat_std,
                    'Flow IAT Max': flow_iat_max,
                    'Flow IAT Min': flow_iat_min,
                    'Fwd IAT Total': sum(flow['fwd_iat']),
                    'Fwd IAT Mean': fwd_iat_mean,
                    'Fwd IAT Std': fwd_iat_std,
                    'Fwd IAT Max': fwd_iat_max,
                    'Fwd IAT Min': fwd_iat_min,
                    'Bwd IAT Total': sum(flow['bwd_iat']),
                    'Bwd IAT Mean': bwd_iat_mean,
                    'Bwd IAT Std': bwd_iat_std,
                    'Bwd IAT Max': bwd_iat_max,
                    'Bwd IAT Min': bwd_iat_min,
                    'Fwd PSH Flags': flag_counts.get('P', 0),
                    'Bwd PSH Flags': flag_counts.get('P', 0) if direction == "backward" else 0,  # 修复
                    'Fwd URG Flags': flag_counts.get('U', 0),
                    'Bwd URG Flags': flag_counts.get('U', 0) if direction == "backward" else 0,  # 修复
                    'Fwd Header Length': sum(flow['fwd_header_lengths']),
                    'Bwd Header Length': sum(flow['bwd_header_lengths']),
                    'Fwd Packets/s': fwd_packets_per_sec,
                    'Bwd Packets/s': bwd_packets_per_sec,
                    'Min Packet Length': all_pl_min,
                    'Max Packet Length': all_pl_max,
                    'Packet Length Mean': all_pl_mean,
                    'Packet Length Std': all_pl_std,
                    'Packet Length Variance': np.var(all_packet_lengths) if all_packet_lengths else 0,
                    'FIN Flag Count': flag_counts.get('F', 0),
                    'SYN Flag Count': flag_counts.get('S', 0),
                    'RST Flag Count': flag_counts.get('R', 0),
                    'PSH Flag Count': flag_counts.get('P', 0),
                    'ACK Flag Count': flag_counts.get('A', 0),
                    'URG Flag Count': flag_counts.get('U', 0),
                    'CWE Flag Count': flag_counts.get('C', 0),  # 修复
                    'ECE Flag Count': flag_counts.get('E', 0),
                    'Down/Up Ratio': total_bwd_bytes / total_fwd_bytes if total_fwd_bytes > 0 else 0,
                    'Average Packet Size': all_pl_mean,
                    'Avg Fwd Segment Size': fwd_pl_mean,
                    'Avg Bwd Segment Size': bwd_pl_mean,
                    'Fwd Header Length.1': fwd_header_mean,  # 重复但保留
                    'Fwd Avg Bytes/Bulk': total_fwd_bytes / max(total_fwd_packets, 1),  # 修复
                    'Fwd Avg Packets/Bulk': 0,  # 简化
                    'Fwd Avg Bulk Rate': 0,  # 简化
                    'Bwd Avg Bytes/Bulk': total_bwd_bytes / max(total_bwd_packets, 1),  # 修复
                    'Bwd Avg Packets/Bulk': 0,  # 简化
                    'Bwd Avg Bulk Rate': 0,  # 简化
                    'Subflow Fwd Packets': total_fwd_packets,
                    'Subflow Fwd Bytes': total_fwd_bytes,
                    'Subflow Bwd Packets': total_bwd_packets,
                    'Subflow Bwd Bytes': total_bwd_bytes,
                    'Init_Win_bytes_forward': flow['fwd_header_lengths'][0] if flow['fwd_header_lengths'] else 0,
                    'Init_Win_bytes_backward': flow['bwd_header_lengths'][0] if flow['bwd_header_lengths'] else 0,
                    'act_data_pkt_fwd': total_fwd_packets,
                    'min_seg_size_forward': min(flow['fwd_packet_lengths']) if flow['fwd_packet_lengths'] else 0,
                    'Active Mean': active_mean,
                    'Active Std': active_std,
                    'Active Max': active_max,
                    'Active Min': active_min,
                    'Idle Mean': idle_mean,
                    'Idle Std': idle_std,
                    'Idle Max': idle_max,
                    'Idle Min': idle_min
                }

                features_list.append(feature)
                metadata_list.append({
                    'src_ip': flow['src_ip'],
                    'dst_ip': flow['dst_ip'],
                    'src_port': flow['src_port'],
                    'dst_port': flow['dst_port'],
                    'protocol': flow['protocol']
                })
            except Exception as e:
                print(f"处理流 {flow_id} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # 创建DataFrame
        features_df = pd.DataFrame(features_list)

        # 确保所有特征都存在
        for feature in CICIDS2017_FEATURES:
            if feature not in features_df.columns:
                features_df[feature] = 0.0
                print(f"警告: 特征 {feature} 未提取，使用默认值0填充")

        # 只保留需要的特征
        features_df = features_df[CICIDS2017_FEATURES]

        # 处理缺失值和无穷大
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.fillna(0, inplace=True)

        # 确保所有特征都是数值类型
        for col in features_df.columns:
            if features_df[col].dtype == 'object':
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                except:
                    features_df[col] = 0.0
            features_df[col] = features_df[col].astype(np.float32)

        print(f"提取CICIDS2017特征数量: {features_df.shape[1]}, 预期特征数量: {len(CICIDS2017_FEATURES)}")

        return features_df, metadata_list

    def extract_features_from_pcap(self, pcap_file):
        """从PCAP文件中提取特征（使用scapy）"""
        if self.dataset_type == "kdd99":
            return self._extract_kdd99_features(pcap_file)
        else:  # CICIDS2017
            return self._extract_cicids2017_features(pcap_file)

    def _extract_kdd99_features(self, pcap_file):
        """为KDD99数据集提取特征"""
        print(f"从 {pcap_file} 提取KDD99特征...")
        packets = rdpcap(pcap_file)
        flows = {}
        metadata_list = []  # 存储流的元数据

        # 遍历所有数据包
        for i, pkt in enumerate(packets):
            # 检查是否有IP层
            if IP not in pkt:
                continue

            # 关键修复：转换时间戳为float
            packet_time = float(pkt.time)

            ip = pkt[IP]
            proto = ip.proto
            src = ip.src
            dst = ip.dst

            # 获取端口信息 - 转换为整数
            sport, dport = 0, 0
            if TCP in pkt:
                transport = pkt[TCP]
                sport = int(transport.sport)
                dport = int(transport.dport)
            elif UDP in pkt:
                transport = pkt[UDP]
                sport = int(transport.sport)
                dport = int(transport.dport)

            # 创建流标识 - 使用整数端口
            flow_id = (src, dst, sport, dport, proto)
            reverse_flow_id = (dst, src, dport, sport, proto)

            # 确定流方向
            if flow_id in flows:
                direction = "forward"
                actual_flow_id = flow_id
            elif reverse_flow_id in flows:
                direction = "backward"
                actual_flow_id = reverse_flow_id
            else:
                direction = "forward"
                actual_flow_id = flow_id
                flows[actual_flow_id] = {
                    'src_ip': src,
                    'dst_ip': dst,
                    'src_port': sport,
                    'dst_port': dport,
                    'protocol': proto,
                    'start_time': packet_time,  # 使用转换后的float
                    'end_time': packet_time,  # 使用转换后的float
                    'fwd_packets': 0,
                    'bwd_packets': 0,
                    'fwd_bytes': 0,
                    'bwd_bytes': 0,
                    'fwd_packet_lengths': [],
                    'bwd_packet_lengths': [],
                    'fwd_iat': [],
                    'bwd_iat': [],
                    'flags': []
                }

            flow = flows[actual_flow_id]
            flow['end_time'] = packet_time  # 使用转换后的float

            if direction == "forward":
                flow['fwd_packets'] += 1
                flow['fwd_bytes'] += len(pkt)
                flow['fwd_packet_lengths'].append(len(pkt))
                if flow['fwd_packets'] > 1:
                    # 计算与前一个数据包的时间间隔
                    interval = packet_time - flow.get('last_fwd_time', packet_time)
                    flow['fwd_iat'].append(float(interval))  # 确保存储为float
                flow['last_fwd_time'] = packet_time
            else:  # 反向流
                flow['bwd_packets'] += 1
                flow['bwd_bytes'] += len(pkt)
                flow['bwd_packet_lengths'].append(len(pkt))
                if flow['bwd_packets'] > 1:
                    interval = packet_time - flow.get('last_bwd_time', packet_time)
                    flow['bwd_iat'].append(float(interval))  # 确保存储为float
                flow['last_bwd_time'] = packet_time

            # 记录TCP标志
            if TCP in pkt:
                tcp = pkt[TCP]
                flags = ""
                if tcp.flags & 0x01: flags += "F"
                if tcp.flags & 0x02: flags += "S"
                if tcp.flags & 0x04: flags += "R"
                if tcp.flags & 0x08: flags += "P"
                if tcp.flags & 0x10: flags += "A"
                if tcp.flags & 0x20: flags += "U"
                if tcp.flags & 0x40: flags += "E"
                if tcp.flags & 0x80: flags += "C"
                flow['flags'].append(flags)

        # 构建特征DataFrame
        features_list = []
        for flow_id, flow in flows.items():
            try:
                # 计算流持续时间（添加最小保护）
                duration = max(flow['end_time'] - flow['start_time'], 1e-6)  # 避免除0

                # 计算统计特征时添加范围限制
                fwd_pl = flow['fwd_packet_lengths'] or [0]
                bwd_pl = flow['bwd_packet_lengths'] or [0]
                fwd_iat = flow['fwd_iat'] or [1e-6]  # 最小时间间隔
                bwd_iat = flow['bwd_iat'] or [1e-6]

                # 安全计算函数（添加范围限制）
                def safe_stat(lst, max_val=1e9):
                    """安全统计函数，限制最大值"""
                    if not lst or len(lst) == 0:
                        return 0, 0, 0, 0

                    # 计算统计量
                    max_val = max_val
                    min_val = min(lst)
                    mean_val = sum(lst) / len(lst)

                    # 方差计算（避免浮点溢出）
                    if len(lst) > 1:
                        var_val = sum((x - mean_val) ** 2 for x in lst) / len(lst)
                        std_val = min(np.sqrt(var_val), max_val)
                    else:
                        std_val = 0

                    # 限制值范围
                    max_val = min(max(lst), max_val)
                    min_val = max(min_val, 0)
                    mean_val = min(mean_val, max_val)
                    std_val = min(std_val, max_val)

                    return max_val, min_val, mean_val, std_val

                # 计算统计量 - 使用Python原生函数
                def safe_max(lst):
                    return max(lst) if lst else 0.0

                def safe_min(lst):
                    return min(lst) if lst else 0.0

                def safe_mean(lst):
                    return sum(lst) / len(lst) if lst else 0.0

                # 使用安全统计函数（设置最大值限制）
                MAX_FLOW_VALUE = 1e9  # 10亿
                fwd_max, fwd_min, fwd_mean, fwd_std = safe_stat(fwd_pl, MAX_FLOW_VALUE)
                bwd_max, bwd_min, bwd_mean, bwd_std = safe_stat(bwd_pl, MAX_FLOW_VALUE)

                # 计算IAT统计量（添加范围限制）
                fwd_iat_max, fwd_iat_min, fwd_iat_mean, fwd_iat_std = safe_stat(fwd_iat, MAX_FLOW_VALUE)
                bwd_iat_max, bwd_iat_min, bwd_iat_mean, bwd_iat_std = safe_stat(bwd_iat, MAX_FLOW_VALUE)

                # 计算流量速率（添加防除0保护）
                total_bytes = flow['fwd_bytes'] + flow['bwd_bytes']
                total_packets = flow['fwd_packets'] + flow['bwd_packets']

                flow_bytes_per_sec = total_bytes / max(duration, 1e-6)
                flow_packets_per_sec = total_packets / max(duration, 1e-6)

                # 限制速率值范围
                flow_bytes_per_sec = min(flow_bytes_per_sec, MAX_FLOW_VALUE)
                flow_packets_per_sec = min(flow_packets_per_sec, MAX_FLOW_VALUE)

                feature = {
                    'duration': duration,
                    'protocol_type': self._map_protocol(flow['protocol']),
                    'service': self._map_service(flow['dst_port']),
                    'flag': self._get_dominant_flag(flow['flags']),
                    'src_bytes': flow['fwd_bytes'],
                    'dst_bytes': flow['bwd_bytes'],
                    'land': 1 if flow['src_ip'] == flow['dst_ip'] and flow['src_port'] == flow['dst_port'] else 0,
                    'wrong_fragment': 0,  # 简化
                    'urgent': 0,  # 简化
                    'hot': 0,  # 简化
                    'num_failed_logins': 0,  # 简化
                    'logged_in': 1 if flow['bwd_packets'] > 0 else 0,
                    'num_compromised': 0,  # 简化
                    'root_shell': 0,  # 简化
                    'su_attempted': 0,  # 简化
                    'num_root': 0,  # 简化
                    'num_file_creations': 0,  # 简化
                    'num_shells': 0,  # 简化
                    'num_access_files': 0,  # 简化
                    'num_outbound_cmds': 0,  # 简化
                    'is_host_login': 0,  # 简化
                    'is_guest_login': 0,  # 简化
                    'count': flow['fwd_packets'] + flow['bwd_packets'],
                    'srv_count': flow['fwd_packets'],
                    'serror_rate': sum(1 for f in flow['flags'] if 'R' in f) / len(flow['flags']) if flow[
                        'flags'] else 0,
                    'srv_serror_rate': 0,  # 简化
                    'rerror_rate': 0,  # 简化
                    'srv_rerror_rate': 0,  # 简化
                    'same_srv_rate': 1.0,  # 简化
                    'diff_srv_rate': 0.0,  # 简化
                    'srv_diff_host_rate': 0.0,  # 简化
                    'dst_host_count': 1,  # 简化
                    'dst_host_srv_count': 1,  # 简化
                    'dst_host_same_srv_rate': 1.0,  # 简化
                    'dst_host_diff_srv_rate': 0.0,  # 简化
                    'dst_host_same_src_port_rate': 0.0,  # 简化
                    'dst_host_srv_diff_host_rate': 0.0,  # 简化
                    'dst_host_serror_rate': 0.0,  # 简化
                    'dst_host_srv_serror_rate': 0.0,  # 简化
                    'dst_host_rerror_rate': 0.0,  # 简化
                    'dst_host_srv_rerror_rate': 0.0,  # 简化
                    # 使用Python原生计算
                    'fwd_packet_length_max': float(fwd_max),
                    'fwd_packet_length_min': float(fwd_min),
                    'fwd_packet_length_mean': float(fwd_mean),
                    'fwd_packet_length_std': float(fwd_std),
                    'bwd_packet_length_max': float(bwd_max),
                    'bwd_packet_length_min': float(bwd_min),
                    'bwd_packet_length_mean': float(bwd_mean),
                    'bwd_packet_length_std': float(bwd_std),
                    'fwd_iat_total': float(sum(fwd_iat)),
                    'fwd_iat_mean': fwd_iat_mean,
                    'fwd_iat_max': fwd_iat_max,
                    'fwd_iat_min': fwd_iat_min,
                    'bwd_iat_total': float(sum(bwd_iat)),
                    'bwd_iat_mean': bwd_iat_mean,
                    'bwd_iat_max': bwd_iat_max,
                    'bwd_iat_min': bwd_iat_min,
                }

                features_list.append(feature)

                # 保存流的元数据
                metadata_list.append({
                    'src_ip': flow['src_ip'],
                    'dst_ip': flow['dst_ip'],
                    'src_port': flow['src_port'],
                    'dst_port': flow['dst_port'],
                    'protocol': flow['protocol']
                })
            except Exception as e:
                print(f"处理流 {flow_id} 时出错: {str(e)}")
                continue

        # 创建DataFrame
        # 使用原始特征名创建DataFrame
        features_df = pd.DataFrame(features_list, columns=self.original_features)

        # 执行与训练时相同的one-hot编码
        features_df = pd.get_dummies(features_df, columns=self.original_categorical_features)

        # 确保特征顺序和数量与训练时一致
        missing_features = set(self.required_features) - set(features_df.columns)
        for feat in missing_features:
            features_df[feat] = 0

        # 只保留训练时使用的特征
        features_df = features_df[self.required_features]

        # 修复：确保所有特征都是数值类型
        # 先尝试转换为数值类型，无法转换的设为0
        features_df = features_df.apply(pd.to_numeric, errors='coerce')

        # 填充任何NaN值为0
        features_df = features_df.fillna(0)

        # 修复：确保所有列都是数值类型
        for col in features_df.columns:
            # 处理空列
            if features_df[col].nunique() == 0:
                features_df[col] = 0.0
                continue

            # 转换非数值列
            if features_df[col].dtype == 'object':
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
                except:
                    le = LabelEncoder()
                    features_df[col] = le.fit_transform(features_df[col])

            # 转换为float32
            features_df[col] = features_df[col].astype(np.float32).fillna(0)

        # 添加特征数量检查
        print(f"提取KDD99特征数量: {features_df.shape[1]}, 预期特征数量: {len(self.required_features)}")

        return features_df, metadata_list  # 返回特征和元数据

    def _map_protocol(self, proto_num):
        """映射协议号到名称"""
        protocol_map = {
            1: 'icmp',
            6: 'tcp',
            17: 'udp'
        }
        return protocol_map.get(proto_num, 'other')

    def _map_service(self, port):
        """映射端口到服务"""
        service_map = {
            80: 'http',
            443: 'https',
            22: 'ssh',
            23: 'telnet',
            21: 'ftp',
            25: 'smtp',
            53: 'domain'
        }
        return service_map.get(port, 'other')

    def _get_dominant_flag(self, flags_list):
        """获取主导TCP标志"""
        if not flags_list:
            return 'OTH'

        flag_counts = {'S': 0, 'A': 0, 'F': 0, 'R': 0, 'P': 0, 'U': 0, 'C': 0, 'E': 0}
        for flags in flags_list:
            for flag in str(flags):
                if flag in flag_counts:
                    flag_counts[flag] += 1

        # 返回出现次数最多的标志
        return max(flag_counts, key=flag_counts.get) if flag_counts else 'OTH'

    def _predict_cicids2017(self, features_df, metadata_list):
        """CIC-IDS2017 专用预测：仅用 mean_/scale_ 手动标准化，绝不调用 scaler.transform。"""
        mean_arr = getattr(self.scaler, "mean_", None)
        scale_arr = getattr(self.scaler, "scale_", None)
        if mean_arr is None or scale_arr is None:
            print("预测过程中发生错误: CIC-IDS2017 scaler 无 mean_/scale_，请删除 ml_models/*_cicids2017_*.pkl 后重新训练")
            return []
        mean_arr = np.atleast_1d(np.asarray(mean_arr, dtype=np.float64)).ravel()
        scale_arr = np.atleast_1d(np.asarray(scale_arr, dtype=np.float64)).ravel()
        n_use = len(mean_arr)
        if n_use == 0:
            return []
        fit_features = getattr(self.scaler, "feature_names_in_", None)
        if fit_features is not None:
            fit_features = list(fit_features)[:n_use]
        else:
            fit_features = (list(self.required_features) if self.required_features else [])[:n_use]
        if not fit_features:
            return []
        for f in set(fit_features) - set(features_df.columns):
            features_df[f] = 0.0
        X = features_df[fit_features].to_numpy(dtype=np.float64, copy=True)
        if X.shape[1] != n_use or len(scale_arr) != n_use:
            print(f"预测过程中发生错误: CIC-IDS2017 特征维度不匹配 (X列={X.shape[1]}, mean={n_use}, scale={len(scale_arr)})")
            return []
        scale_safe = np.where(scale_arr != 0, scale_arr, 1.0)
        scaled_features = (X - mean_arr) / scale_safe
        # 仅用 numpy 做预测，不经过 scaler
        if self.model_type == "isolation_forest":
            scores = self.model.predict(scaled_features)
            predictions = ["anomaly" if s == -1 else "normal" for s in scores]
            confidences = [float(abs(s)) for s in scores]
            top3_classes = [[]] * len(predictions)
            top3_confidences = [[]] * len(predictions)
        else:
            predictions = self.model.predict(scaled_features)
            try:
                confidences = np.max(self.model.predict_proba(scaled_features), axis=1)
                top3_probs = self.model.predict_proba(scaled_features)
                top3_classes = []
                top3_confidences = []
                for i in range(top3_probs.shape[0]):
                    idx = np.argsort(top3_probs[i])[-3:][::-1]
                    classes = [self.label_encoder.classes_[j] if j < len(self.label_encoder.classes_) else f"未知_{j}" for j in idx]
                    top3_classes.append(classes)
                    top3_confidences.append(top3_probs[i][idx].tolist())
            except Exception:
                confidences = [0.9] * len(predictions)
                top3_classes = [["未知"]] * len(predictions)
                top3_confidences = [[0.0]] * len(predictions)
        readable_predictions = []
        for pred in predictions:
            try:
                # 分类模型返回整数索引，需用 label_encoder 解码为类别名再映射（如 0 -> 'normal' -> 正常流量）
                if self.label_encoder and isinstance(pred, (int, np.integer)) and 0 <= int(pred) < len(self.label_encoder.classes_):
                    raw_label = self.label_encoder.classes_[int(pred)]
                    r = self.ATTACK_TYPE_MAPPING[self.dataset_type].get(raw_label, f"未知攻击类型 ({pred})")
                else:
                    r = self.ATTACK_TYPE_MAPPING[self.dataset_type].get(pred, f"未知攻击类型 ({pred})")
            except Exception:
                r = str(pred)
            readable_predictions.append(r)
        results = []
        for i in range(len(features_df)):
            meta = metadata_list[i] if i < len(metadata_list) else {'src_ip': '未知', 'dst_ip': '未知', 'src_port': 0, 'dst_port': 0, 'protocol': 0}
            res = {
                "src_ip": meta['src_ip'], "dst_ip": meta['dst_ip'], "src_port": meta['src_port'],
                "dst_port": meta['dst_port'], "protocol": self._map_protocol(meta['protocol']),
                "prediction": readable_predictions[i], "confidence": float(confidences[i]),
                "model_type": self.model_type, "dataset": self.dataset_type, "attack_details": []
            }
            if i < len(top3_classes):
                for j, raw in enumerate(top3_classes[i]):
                    conf = float(top3_confidences[i][j]) if j < len(top3_confidences[i]) else 0.0
                    if raw == "未知" or (isinstance(raw, str) and "未知" in raw):
                        continue
                    at = self.ATTACK_TYPE_MAPPING[self.dataset_type].get(raw, raw)
                    if at and at != "未知" and "未知" not in str(at):
                        res["attack_details"].append({"type": at, "confidence": conf})
            results.append(res)
        return results

    def predict(self, pcap_file):
        try:
            # 【必须最先】CIC-IDS2017 走独立分支，绝不调用 scaler.transform
            if self.dataset_type == "cicids2017":
                features_df, metadata_list = self.extract_features_from_pcap(pcap_file)
                if features_df.empty:
                    print("警告：CIC-IDS2017 未提取到任何特征")
                    return []
                return self._predict_cicids2017(features_df, metadata_list)

            # 以下仅 KDD99
            features_df, metadata_list = self.extract_features_from_pcap(pcap_file)
            if features_df.empty:
                print("警告：未提取到任何特征")
                return []

            # 以下为 KDD99 或通用路径（CIC-IDS2017 已在上方 return）
            # CIC-IDS2017：仅用 mean_/scale_ 手动标准化，绝不调用 scaler.transform，避免 sklearn 列名校验
            mean_arr = getattr(self.scaler, "mean_", None)
            scale_arr = getattr(self.scaler, "scale_", None)
            if mean_arr is not None:
                mean_arr = np.atleast_1d(np.asarray(mean_arr, dtype=np.float64)).ravel()
            if scale_arr is not None:
                scale_arr = np.atleast_1d(np.asarray(scale_arr, dtype=np.float64)).ravel()
            n_use = int(len(mean_arr)) if mean_arr is not None and len(mean_arr) > 0 else None
            if n_use is None:
                n_use = getattr(self.scaler, "n_features_in_", None)
            if n_use is None:
                n_use = len(self.required_features) if self.required_features else 0

            fit_features = getattr(self.scaler, "feature_names_in_", None)
            if fit_features is not None:
                fit_features = list(fit_features)
            else:
                fit_features = list(self.required_features) if self.required_features else []
            if n_use <= 0 or not fit_features:
                print(f"预测过程中发生错误: 无法获取训练时的特征列表，请删除 ml_models/*_{self.dataset_type}_*.pkl 后重新分析以触发重新训练")
                return []
            if len(fit_features) > n_use:
                fit_features = fit_features[:n_use]
            missing_features = set(fit_features) - set(features_df.columns)
            for feat in missing_features:
                features_df[feat] = 0.0
            ordered = features_df[fit_features]
            X = ordered.to_numpy(dtype=np.float64, copy=True)
            if X.ndim != 2 or X.shape[1] != n_use:
                print(f"预测过程中发生错误: 特征矩阵形状异常 shape={X.shape}，期望列数={n_use}")
                return []

            # 仅手动标准化，不调用 scaler.transform
            if mean_arr is not None and scale_arr is not None and len(mean_arr) == X.shape[1] and len(scale_arr) == X.shape[1]:
                scale_safe = np.where(scale_arr != 0, scale_arr, 1.0)
                scaled_features = (X - mean_arr) / scale_safe
            else:
                print(f"预测过程中发生错误: scaler 维度与特征数不匹配(mean={len(mean_arr) if mean_arr is not None else 0}, scale={len(scale_arr) if scale_arr is not None else 0}, X列={X.shape[1]})，请删除 ml_models/*_{self.dataset_type}_*.pkl 后重新训练")
                return []

            # 预测
            if self.model_type == "isolation_forest":
                # 异常检测模型返回异常分数（-1表示异常）
                scores = self.model.predict(scaled_features)
                predictions = ["anomaly" if score == -1 else "normal" for score in scores]
                confidences = [abs(score) for score in scores]  # 使用绝对距离作为置信度

                # 对于Isolation Forest，仅能判断异常/正常，无具体攻击类型，不填充“未知”占位
                top3_classes = [[]] * len(predictions)
                top3_confidences = [[]] * len(predictions)
            else:
                # 分类模型返回预测类别和概率
                predictions = self.model.predict(scaled_features)

                # 尝试获取概率，如果失败则使用默认置信度
                try:
                    confidences = np.max(self.model.predict_proba(scaled_features), axis=1)
                    # 获取预测概率最高的前3个类别
                    top3_probs = self.model.predict_proba(scaled_features)

                    # 修复：单独处理每个样本的前3个预测
                    top3_classes = []
                    top3_confidences = []

                    for i in range(top3_probs.shape[0]):
                        # 获取当前样本的预测概率
                        sample_probs = top3_probs[i]

                        # 获取前3个最高概率的索引
                        top3_idx = np.argsort(sample_probs)[-3:][::-1]  # 从高到低排序

                        # 安全地转换索引为类别名称
                        classes = []
                        for idx in top3_idx:
                            # 检查索引是否在有效范围内
                            if idx < len(self.label_encoder.classes_):
                                classes.append(self.label_encoder.classes_[idx])
                            else:
                                # 创建替代标签
                                classes.append(f"未知标签_{idx}")

                        # 获取对应的置信度
                        confs = sample_probs[top3_idx]

                        top3_classes.append(classes)
                        top3_confidences.append(confs.tolist())

                except AttributeError:
                    confidences = [0.9] * len(predictions)
                    top3_classes = [["未知"]] * len(predictions)
                    top3_confidences = [[0.0]] * len(predictions)

            readable_predictions = []

            # 解码主要预测标签：分类模型返回整数索引，需用 label_encoder.classes_[pred] 解码为类别名再映射
            for pred in predictions:
                try:
                    if self.label_encoder and isinstance(pred, (int, np.integer)) and 0 <= int(pred) < len(self.label_encoder.classes_):
                        raw_label = self.label_encoder.classes_[int(pred)]
                        readable = self.ATTACK_TYPE_MAPPING[self.dataset_type].get(raw_label, f"未知攻击类型 ({pred})")
                    else:
                        readable = self.ATTACK_TYPE_MAPPING[self.dataset_type].get(pred, f"未知攻击类型 ({pred})")
                except Exception:
                    readable = str(pred)
                readable_predictions.append(readable)

            # 准备结果
            results = []
            for i in range(len(features_df)):
                # 获取元数据
                meta = metadata_list[i] if i < len(metadata_list) else {
                    'src_ip': '未知',
                    'dst_ip': '未知',
                    'src_port': 0,
                    'dst_port': 0,
                    'protocol': 0
                }

                result = {
                    "src_ip": meta['src_ip'],
                    "dst_ip": meta['dst_ip'],
                    "src_port": meta['src_port'],
                    "dst_port": meta['dst_port'],
                    "protocol": self._map_protocol(meta['protocol']),
                    "prediction": readable_predictions[i],
                    "confidence": float(confidences[i]),
                    "model_type": self.model_type,
                    "dataset": self.dataset_type,
                    "attack_details": []
                }

                # 添加前3个可能的攻击类型（跳过“未知”占位，不展示无信息类型）
                if i < len(top3_classes):
                    for j in range(len(top3_classes[i])):
                        raw = top3_classes[i][j]
                        conf = float(top3_confidences[i][j]) if j < len(top3_confidences[i]) else 0.0
                        if raw == "未知" or (isinstance(raw, str) and "未知" in raw):
                            continue
                        attack_type = self.ATTACK_TYPE_MAPPING[self.dataset_type].get(
                            raw, raw
                        )
                        if attack_type == "未知" or (isinstance(attack_type, str) and "未知" in attack_type):
                            continue
                        result["attack_details"].append({
                            "type": attack_type,
                            "confidence": conf
                        })

                results.append(result)

            return results
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"预测过程中发生错误: {str(e)}")
            return []


# ==================== 集成规则匹配和ML引擎 ====================
class AnomalyDetector:
    def __init__(self):
        self.rule_manager = None
        self.ml_engines = {}

        # 确保目录存在
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

    def initialize_ml_engines(self):
        """初始化所有机器学习引擎"""
        for model_type in MODEL_TYPES:
            for dataset_type in ["kdd99", "cicids2017"]:
                key = f"{model_type}_{dataset_type}"
                try:
                    self.ml_engines[key] = MLEngine(model_type, dataset_type)
                except Exception as e:
                    print(f"初始化ML引擎 {key} 失败: {str(e)}")

    def analyze_pcap(self, pcap_file):
        """分析PCAP文件"""
        start_time = time.time()

        # 2. 机器学习分析
        ml_results = []
        for key, engine in self.ml_engines.items():
            try:
                engine_start = time.time()
                predictions = engine.predict(pcap_file)
                proc_time = time.time() - engine_start

                # 添加处理时间信息
                for pred in predictions:
                    pred["processing_time"] = proc_time

                ml_results.extend(predictions)
                print(
                    f"{key} 分析完成: 检测到 {len([p for p in predictions if p['prediction'] != '正常流量'])} 个异常事件")
            except Exception as e:
                print(f"{key} 分析失败: {str(e)}")

        # 3. 整合结果
        combined_results = {
            "ml_based_predictions": ml_results,
            "summary": {
                "total_ml_anomalies": sum(1 for p in ml_results if p['prediction'] != '正常流量'),
                "analysis_time": time.time() - start_time
            }
        }

        return combined_results

    def save_results(self, results, output_file):
        """保存检测结果"""
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存至 {output_file}")

    def run(self, pcap_file):
        """运行完整分析流程"""
        start_time = time.time()

        # 初始化ML引擎
        self.initialize_ml_engines()

        # 分析PCAP文件
        results = self.analyze_pcap(pcap_file)

        # 保存结果
        self.save_results(results, OUTPUT_FILE)

        # 打印摘要
        print("\n分析摘要:")
        print(f"总处理时间: {time.time() - start_time:.2f} 秒")
        print(f"机器学习异常检测: {results['summary']['total_ml_anomalies']}")


# ==================== 数据集下载函数 ====================
def download_datasets():
    """下载网络安全数据集"""
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # KDD99数据集
    kdd_path = os.path.join(DATASETS_DIR, "kddcup.data_10_percent.gz")
    if not os.path.exists(kdd_path):
        print("下载KDD99数据集...")
        url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
        try:
            urlretrieve(url, kdd_path)
            print("KDD99数据集下载完成")
        except Exception as e:
            print(f"KDD99数据集下载失败: {str(e)}")
    else:
        print("KDD99数据集已存在")

    # CICIDS2017数据集（需要手动下载）
    cicids_dir = os.path.join(DATASETS_DIR, "MachineLearningCVE")
    if not os.path.exists(cicids_dir):
        print("\nCICIDS2017数据集需要手动下载:")
        print("1. 访问: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("2. 下载 'CSV files' 和 'PCAP files'")
        print("3. 解压到: datasets/MachineLearningCVE/")
        print("4. 确保包含以下文件:")
        print("   - Monday-WorkingHours.pcap")
        print("   - Tuesday-WorkingHours.pcap")
        print("   - Wednesday-WorkingHours.pcap")
        print("   - Thursday-WorkingHours.pcap")
        print("   - Friday-WorkingHours.pcap")
        print("   - 以及对应的CSV文件\n")


# ==================== 主函数 ====================
if __name__ == "__main__":
    # 下载数据集（指导用户操作）
    download_datasets()

    # 创建分析器
    analyzer = AnomalyDetector()

    # 运行分析
    analyzer.run(PCAP_FILE)