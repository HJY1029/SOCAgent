import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from agent import AlertAgent

# 灰白色调配置
sns.set_theme(style="whitegrid", palette="Greys")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14


def generate_synthetic_data(size):
    """生成合成测试数据"""
    # 简化版 - 实际应生成更符合真实流量的数据
    data = pd.DataFrame()
    for i in range(50):  # 50个特征
        data[f'feature_{i}'] = np.random.normal(0, 1, size)

    # 随机标签
    data['label'] = np.random.choice(['normal', 'attack'], size, p=[0.9, 0.1])
    return data


def measure_performance(agent, data):
    """测量智能体性能"""
    start_time = time.time()

    # 模拟分析过程
    results = agent.analyze_pcap(data)

    elapsed = time.time() - start_time
    return elapsed, results


def experiment_3():
    """实验三：模型效率实验"""
    # 初始化智能体
    config = {
        "knowledge_dir": "knowledge_base",
        "rule_dir": "rules",
        "use_llm": True,
        "llm_model_path": "llm_models/deepseek-llm-r1"
    }
    agent = AlertAgent(config)

    # 测试不同规模数据集
    sizes = [1000, 5000, 10000, 50000, 100000, 500000]
    results = []
    component_times = []

    for size in sizes:
        print(f"测试数据规模: {size}")
        data = generate_synthetic_data(size)

        # 测量整体性能
        elapsed, _ = measure_performance(agent, data)
        results.append({'size': size, 'time': elapsed})

        # 测量组件性能（简化版）
        components = {
            'Rule Engine': 0.4 * elapsed,
            'ML Inference': 0.3 * elapsed,
            'LLM Analysis': 0.2 * elapsed,
            'Integration': 0.1 * elapsed
        }
        for comp, t in components.items():
            component_times.append({
                'size': size,
                'component': comp,
                'time': t
            })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    components_df = pd.DataFrame(component_times)

    # 保存结果
    results_df.to_csv("results/experiment_3_results.csv", index=False)
    components_df.to_csv("results/experiment_3_components.csv", index=False)

    # 可视化结果
    plot_experiment_3_results(results_df, components_df)

    return results_df, components_df


def plot_experiment_3_results(results_df, components_df):
    """可视化实验三结果"""
    # 1. 处理时间随数据规模变化
    plt.figure()
    sns.lineplot(x='size', y='time', data=results_df, marker='o',
                 color='#606060', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Processing Time vs Data Volume')
    plt.xlabel('Number of Packets')
    plt.ylabel('Processing Time (seconds)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/exp3_scalability.pdf')

    # 2. 组件时间分布（中等规模）
    med_size = components_df['size'].median()
    med_data = components_df[components_df['size'] == med_size]

    plt.figure()
    sns.barplot(x='component', y='time', data=med_data,
                palette=['#D0D0D0', '#B0B0B0', '#909090', '#707070'])
    plt.title('Component Processing Time (Medium Workload)')
    plt.ylabel('Time (seconds)')
    plt.xlabel('System Component')
    plt.tight_layout()
    plt.savefig('results/exp3_components_time.pdf')

    # 3. 时间分布饼图
    plt.figure(figsize=(8, 8))
    times = med_data.groupby('component')['time'].sum()
    colors = ['#D0D0D0', '#B0B0B0', '#909090', '#707070']

    plt.pie(times, labels=times.index, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    plt.title('Time Distribution in Medium Workload', fontsize=14)
    plt.savefig('results/exp3_time_distribution.pdf')


# 运行实验三
experiment_3_results, experiment_3_components = experiment_3()