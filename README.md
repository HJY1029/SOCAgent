# SOCAgent · 基于大模型的安全告警智能研判系统

基于 PCAP 的流量安全分析平台：上传流量包，由后端进行 Snort 规则检测、机器学习辅助检测与智能研判，并返回分析报告。

> 本仓库托管于 [GitHub · HJY1029/SOCAgent](https://github.com/HJY1029/SOCAgent)。

## 功能

- 支持 `.pcap` / `.pcapng` / `.cap` 上传
- Snort 规则引擎检测
- 知识库检索与智能研判
- Web 界面（科技风 UI）查看分析结果

## 克隆仓库（请先读这一段）

本项目的**训练数据集**和**预训练模型**体积较大（合计约 900MB），已通过 **Git LFS** 存放在 GitHub 上。若只执行普通 `git clone`，你看到的 CSV / `.pkl` 可能只有一百多字节的「指针文件」，并不是真实数据，机器学习相关功能会无法正常运行。

**推荐步骤：**

1. 安装 [Git](https://git-scm.com/) 与 [Git LFS](https://git-lfs.com/)（Windows 用户安装 Git 时勾选 LFS 即可）。
2. 克隆并拉取大文件：

```bash
git clone https://github.com/HJY1029/SOCAgent.git
cd SOCAgent
git lfs install
git lfs pull
```

若你已经克隆过仓库，在项目目录补拉 LFS 文件：

```bash
git pull
git lfs pull
```

**如何确认下载成功？** 打开 `engine/datasets/MachineLearningCVE/`，最大的 `Wednesday-workingHours.pcap_ISCX.csv` 应约为 **215MB**；若只有一百多字节，说明 LFS 尚未拉取完成，请重试 `git lfs pull`。

**仓库里的大文件包括：**

| 路径 | 内容 | 说明 |
|------|------|------|
| `engine/datasets/` | CIC-IDS2017 CSV、KDD Cup 样本 | 用于训练 / 评估 ML 引擎 |
| `engine/ml_models/*.pkl` | 预训练模型与编码器 | 开箱即用，无需重新训练即可体验 |

免费 GitHub 账户的 LFS 存储与流量有配额限制；若 `git lfs pull` 失败，可多试几次，或检查网络与代理设置。

---

## 快速开始

### 1. 依赖

```bash
pip install -r requirements-web.txt
# 以及项目其他依赖（见项目根目录其他 requirements 或环境）
```

### 2. 启动后端

```bash
python app.py
```

- 后端地址：<http://127.0.0.1:5000/>
- 热重载已开启，修改 `.py` 后会自动重启

### 3. 使用前端

浏览器访问 **http://127.0.0.1:5000/**，选择或拖入 PCAP 文件，点击「开始分析」，在页面查看报告。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | 首页（静态） |
| GET  | `/api/health` | 健康检查 |
| POST | `/api/upload` | 上传文件（表单字段 `file`），返回 `{ success, report? }` 或 `{ success: false, error }` |

## 项目结构（简要）

```
SOCAgent/
├── README.md              # 本文件
├── app.py                 # Flask 后端入口
├── agent.py               # 智能研判智能体
├── security_agent.py      # 安全分析智能体
├── knowledge_retriever.py   # 知识库检索
├── static/
│   └── index.html         # 前端页面
├── engine/
│   ├── ids_analyzer.py    # Snort 规则与 PCAP 分析
│   ├── ml_engine.py       # 机器学习检测引擎
│   ├── rules/             # Snort 规则（.rules）
│   ├── datasets/          # 数据集（Git LFS，见上文）
│   └── ml_models/         # 预训练模型（Git LFS）
├── uploads/               # 上传 PCAP 保存目录
└── requirements-web.txt
```

## 常见问题

- **克隆后 ML 功能报错或找不到模型？** 先执行 `git lfs pull`，确认 `engine/ml_models/` 下 `.pkl` 文件大小正常（最大的 LightGBM 模型约 10MB）。
- **规则目录为空？** `engine/rules` 若不存在，程序会在首次加载时尝试自动创建并下载规则。
- **上传失败？** 默认单文件上限 200MB，可在 `app.py` 中修改 `MAX_CONTENT_LENGTH`。

## 说明

- 大文件跟踪规则见仓库根目录 `.gitattributes`。
- 上传文件保存在 `uploads/`，该目录已在 `.gitignore` 中忽略，不会进入版本库。
