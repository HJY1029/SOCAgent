import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, roc_auc_score
from engine.ml_engine import MLEngine, KDD99_FEATURES  # 假设我们的机器学习引擎已实现
import time
import os

# 灰白色调配置
sns.set_theme(style="whitegrid", palette="Greys")
plt.rcParams['axes.titlepad'] = 15
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (10, 6)


def load_dataset(dataset_type):
    """加载KDD99或CICIDS2017数据集"""
    if dataset_type == "kdd99":
        # 这里简化了实际的数据加载过程
        data = pd.read_csv(f"datasets/kddcup.data_10_percent.gz", header=None)
        data.columns = KDD99_FEATURES + ['label']
        # 简化标签为二分类：正常 vs 攻击
        data['label'] = data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
        return data
    else:  # CICIDS2017
        data = pd.concat([pd.read_csv(f) for f in glob.glob("datasets/MachineLearningCVE/*.csv")])
        data.rename(columns={'Label': 'label'}, inplace=True)
        # 简化标签
        data['label'] = data['label'].apply(lambda x: 'BENIGN' if x == 'BENIGN' else 'attack')
        return data


def evaluate_model(model_type, dataset_type, X_train, y_train, X_test, y_test):
    """评估单个模型的性能"""
    try:
        # 初始化并训练模型
        model = MLEngine(model_type, dataset_type)

        # 评估模型
        y_pred = model.model.predict(model.scaler.transform(X_test))

        if model_type == "isolation_forest":
            # 转换异常检测输出
            y_pred = ["attack" if p == -1 else "normal" for p in y_pred]

        # 计算各项指标
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred, pos_label="attack")
        recall = recall_score(y_test, y_pred, pos_label="attack")
        precision = precision_score(y_test, y_pred, pos_label="attack")

        # 计算AUC需要二值化
        y_test_bin = [1 if l == "attack" else 0 for l in y_test]
        y_pred_bin = [1 if p == "attack" else 0 for p in y_pred]
        auc = roc_auc_score(y_test_bin, y_pred_bin)

        return {
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'auc': auc,
            'report': report
        }
    except Exception as e:
        print(f"评估模型 {model_type} 失败: {str(e)}")
        return None


def experiment_1():
    """实验一：异常检测准确性实验"""
    results = []
    datasets = ["kdd99", "cicids2017"]
    models = ["isolation_forest", "xgboost", "lightgbm", "random_forest"]

    for dataset_type in datasets:
        # 加载数据集
        data = load_dataset(dataset_type)

        # 数据预处理
        X = data.drop('label', axis=1)
        y = data['label']

        # 只保留数值型特征
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(0)

        # 分层K折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for model_type in models:
                start_time = time.time()
                metrics = evaluate_model(model_type, dataset_type, X_train, y_train, X_test, y_test)
                elapsed = time.time() - start_time

                if metrics:
                    results.append({
                        'dataset': dataset_type,
                        'model': model_type,
                        'fold': fold,
                        'f1': metrics['f1'],
                        'recall': metrics['recall'],
                        'precision': metrics['precision'],
                        'auc': metrics['auc'],
                        'time': elapsed
                    })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存结果
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/experiment_1_results.csv", index=False)

    # 可视化结果
    plot_experiment_1_results(results_df)

    return results_df


def plot_experiment_1_results(df):
    """可视化实验一结果"""
    # 1. 模型性能对比（F1分数）
    plt.figure()
    sns.barplot(x='model', y='f1', hue='dataset', data=df,
                palette=['#808080', '#606060'], errorbar='sd')
    plt.title('F1-Score Comparison Across Models and Datasets')
    plt.ylabel('F1-Score')
    plt.xlabel('Detection Models')
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig('results/exp1_f1_comparison.pdf')

    # 2. 召回率和精确率对比
    plt.figure()
    melted_df = pd.melt(df, id_vars=['model', 'dataset'],
                        value_vars=['recall', 'precision'],
                        var_name='metric', value_name='score')

    sns.barplot(x='model', y='score', hue='metric', data=melted_df,
                palette=['#A0A0A0', '#707070'])
    plt.title('Recall and Precision Comparison')
    plt.ylabel('Score')
    plt.xlabel('Detection Models')
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig('results/exp1_recall_precision.pdf')

    # 3. 模型效率对比
    plt.figure()
    sns.barplot(x='model', y='time', data=df, color='#909090')
    plt.title('Model Inference Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Detection Models')
    plt.tight_layout()
    plt.savefig('results/exp1_inference_time.pdf')

    # 4. AUC对比
    plt.figure()
    sns.barplot(x='model', y='auc', hue='dataset', data=df,
                palette=['#808080', '#606060'], errorbar='sd')
    plt.title('AUC-ROC Comparison')
    plt.ylabel('AUC Score')
    plt.xlabel('Detection Models')
    plt.ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig('results/exp1_auc_comparison.pdf')


# 运行实验一
experiment_1_results = experiment_1()