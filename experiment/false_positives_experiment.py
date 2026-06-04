import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score
from engine.ml_engine import MLEngine
import random

from experiment.accuracy_experiment import load_dataset

# 灰白色调配置
sns.set_theme(style="whitegrid", palette="Greys")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14


def generate_false_positives(data, ratio=0.3):
    """生成误报数据集"""
    # 只选择正常样本
    normal_samples = data[data['label'] == 'normal']

    # 随机选择一部分正常样本
    n_fp = int(len(normal_samples) * ratio)
    fp_samples = normal_samples.sample(n_fp, random_state=42).copy()

    # 修改标签为攻击，创建误报
    fp_samples['label'] = 'attack'

    # 添加轻微噪声
    for col in fp_samples.columns:
        if col != 'label' and pd.api.types.is_numeric_dtype(fp_samples[col]):
            noise = np.random.normal(0, 0.1 * fp_samples[col].std(), size=len(fp_samples))
            fp_samples[col] += noise

    return fp_samples


def evaluate_false_positives(model, X, y_true):
    """评估模型在误报数据上的表现"""
    # 模型预测
    y_pred = model.model.predict(model.scaler.transform(X))

    if isinstance(model.model, IsolationForest):
        y_pred = ["attack" if p == -1 else "normal" for p in y_pred]

    # 计算纠正率（模型预测为正常的比例）
    corrected = sum([1 for p, t in zip(y_pred, y_true) if p == 'normal' and t == 'attack'])
    correction_rate = corrected / len(y_true)

    # 计算精确率
    precision = precision_score(y_true, y_pred, pos_label="attack")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=['normal', 'attack'])

    return correction_rate, precision, cm


def experiment_2():
    """实验二：误报检测实验"""
    results = []
    datasets = ["kdd99", "cicids2017"]
    models = ["isolation_forest", "xgboost", "lightgbm", "random_forest"]
    scenarios = ["base", "adversarial", "hybrid"]

    for dataset_type in datasets:
        # 加载数据集
        data = load_dataset(dataset_type)

        # 数据预处理
        X = data.drop('label', axis=1).select_dtypes(include=[np.number]).fillna(0)
        y = data['label']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        for model_type in models:
            # 训练模型
            model = MLEngine(model_type, dataset_type)

            for scenario in scenarios:
                # 生成误报数据
                if scenario == "base":
                    fp_data = generate_false_positives(
                        pd.concat([X_test, y_test], axis=1), ratio=0.3
                    )
                elif scenario == "adversarial":
                    fp_data = generate_false_positives(
                        pd.concat([X_test, y_test], axis=1), ratio=0.3
                    )
                    # 添加更强的对抗性扰动
                    for col in fp_data.columns:
                        if col != 'label' and pd.api.types.is_numeric_dtype(fp_data[col]):
                            noise = np.random.normal(0, 0.3 * fp_data[col].std(), size=len(fp_data))
                            fp_data[col] += noise
                else:  # hybrid
                    fp_data = generate_false_positives(
                        pd.concat([X_test, y_test], axis=1), ratio=0.2
                    )
                    # 添加真实攻击样本但标记为正常
                    attack_samples = data[data['label'] == 'attack'].sample(100, random_state=42)
                    attack_samples['label'] = 'normal'
                    fp_data = pd.concat([fp_data, attack_samples])

                X_fp = fp_data.drop('label', axis=1)
                y_fp = fp_data['label']

                # 评估误报检测
                correction_rate, precision, cm = evaluate_false_positives(model, X_fp, y_fp)

                results.append({
                    'dataset': dataset_type,
                    'model': model_type,
                    'scenario': scenario,
                    'correction_rate': correction_rate,
                    'precision': precision,
                    'confusion_matrix': cm
                })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/experiment_2_results.csv", index=False)

    # 可视化结果
    plot_experiment_2_results(results_df)

    return results_df


def plot_experiment_2_results(df):
    """可视化实验二结果"""
    # 1. 误报纠正率对比
    plt.figure()
    sns.barplot(x='model', y='correction_rate', hue='scenario', data=df,
                palette=['#404040', '#808080', '#C0C0C0'])
    plt.title('False Positive Correction Rate')
    plt.ylabel('Correction Rate')
    plt.xlabel('Detection Models')
    plt.ylim(0, 1.0)
    plt.legend(title='Scenario')
    plt.tight_layout()
    plt.savefig('results/exp2_correction_rate.pdf')

    # 2. 不同场景下精确率对比
    plt.figure()
    sns.barplot(x='scenario', y='precision', hue='model', data=df,
                palette=['#404040', '#707070', '#A0A0A0', '#D0D0D0'])
    plt.title('Precision in Different False Positive Scenarios')
    plt.ylabel('Precision')
    plt.xlabel('False Positive Scenario')
    plt.ylim(0.7, 1.0)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig('results/exp2_precision_scenarios.pdf')

    # 3. 混淆矩阵示例（最后一个模型的）
    last_result = df.iloc[-1]
    cm = last_result['confusion_matrix']
    labels = ['Normal', 'Attack']

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix on False Positive Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/exp2_confusion_matrix.pdf')


# 运行实验二
experiment_2_results = experiment_2()