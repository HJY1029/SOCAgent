# Security Analysis · 流量安全分析

基于 PCAP 的流量安全分析平台：上传流量包，由后端进行 Snort 规则检测与智能研判，并返回分析报告。

## 功能

- 支持 `.pcap` / `.pcapng` / `.cap` 上传
- Snort 规则引擎检测
- 知识库检索与智能研判
- Web 界面（科技风 UI）查看分析结果

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
securityAnalysis/
├── README.md           # 本文件
├── app.py              # Flask 后端入口
├── agent.py            # 智能研判智能体
├── knowledge_retriever.py
├── static/
│   └── index.html      # 前端页面
├── engine/
│   ├── ids_analyzer.py # Snort 规则与 PCAP 分析
│   ├── rules/          # Snort 规则文件（.rules）
│   └── ...
├── uploads/            # 上传文件保存目录
└── requirements-web.txt
```

## 说明

- 规则目录 `engine/rules` 若不存在会在首次加载时自动创建并可下载规则。
- 上传文件大小限制 200MB，可在 `app.py` 中修改 `MAX_CONTENT_LENGTH`。
