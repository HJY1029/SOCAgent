**1** **INTRODUCTION**

Digitalization highlights cybersecurity’s importance, and SOC—defense’s central nervous system—relies on alert detection/assessment for effective threat response. Verizon report [1] analyzed 30,458 security incidents and 10,626 confirmed breaches across 94 countries, while China [2] is among the most severely affected by cybersecurity threats, with key infrastructure facing persistent attack risks. Halliburton's $35 million loss from a ransomware attack [3] underscores SOC urgency, with alert analysis as a key threat identification link.

Traditional analysis has bottlenecks: manual judgment depends on expert experience, leading to false alarms, delayed responses, and high talent costs [4]; fixed engines like SOAR rely on preset scripts, failing to adapt to unknown threats/dynamic environments with high maintenance costs [10].

LLM’s breakthrough in semantic understanding and RAG’s external knowledge integration offer solutions. This paper proposes an LLM-RAG enhanced dynamic workflow engine framework, transforming fixed execution into intelligent orchestration via process extraction, engine architecture design, and dynamic judgment path generation.

# 2 Related Work

Network attack alert analysis and triage has evolved from manual assistance toward intellectualization and automation. The research community generally focuses on detection efficiency, accuracy, and adaptability to emerging threats. This section reviews existing work by categorizing them into traditional alert analysis methods and intelligent alert analysis technologies, combined with the content of the references.

## 2.1 Research on Traditional Network Attack Alert Analysis Methods

### 2.1.1 Manual Analysis Assistance and Host/Anomaly Detection

Manual triage suffers from information overload and over-reliance on experience. In host and anomaly detection, Bejtlich [6] proposed the NSM operational framework in *The Tao of Network Security Monitoring: Beyond Intrusion Detection*, and introduced methods for mining network traffic data using open-source tools to optimize monitoring and incident response. However, the false positive rate remains high in practice.

Hassan et al. [7] proposed NoDoze, which uses causal dependency graphs for automated alert triage and root cause analysis: it assigns anomaly scores based on historical frequency and propagates scores via graph diffusion to generate prioritized subgraphs for analysts. Evaluation on 364 threat alerts shows an approximately 86% reduction in false positives and significant time savings. Nevertheless, the method relies on causal graph construction, and its portability across different SOC environments is limited by graph quality and data availability.

### 2.1.2 Fixed Detection Engines and SOAR Systems

Traditional IDS and fixed-rule engines have clear limitations. Sommer and Paxson [8] pointed out that traditional network intrusion detection based on the “closed-world” assumption struggles with unknown attacks, and discussed opportunities and challenges of applying machine learning to network IDS.

For anomaly detection algorithms, Liu et al. [9] proposed Isolation Forest, which achieves efficient anomaly detection through random sub-sampling and isolation trees, and is widely used in point anomaly identification. However, its generalization and interpretability in complex attack scenarios remain limited.

Bridges et al. [10] conducted the first large-scale empirical user study on six commercial SOAR tools. In a constructed cyber range, experiments with 24 security operators verified that SOAR tools significantly improve security investigation efficiency and reduce analyst context switching, but decrease the accuracy and completeness of ticket records. The practical performance of SOAR highly depends on tool configuration, and excessive automation may degrade analysis quality. This work empirically validates the trade-offs among usability, automation level, and task quality in SOAR, providing a practical basis for designing adaptive response mechanisms for real-world security operations in this study.

## 2.2 Research Status of Intelligent Alert Analysis Technologies

### 2.2.1 Machine Learning-Driven False Positive Filtering and Threat Classification

Machine learning is widely used for false positive filtering and threat classification in alert analysis. Bhati et al. [11] applied an XGBoost-based ensemble method for network intrusion detection, achieving a better bias-variance trade-off and improving the efficiency and accuracy of ensemble IDS.

Hou et al. [12] proposed a hierarchical LSTM network for network attack detection from traffic, enhancing the modeling of attack patterns via sequential modeling. Huang et al. [13] combined graph neural networks (GNN) with cross-protocol analysis to detect malicious IPs, achieving high accuracy in malicious IP identification. These works mainly optimize single-stage detection or classification, lacking orchestration and scheduling for the full alert workflow, which distinguishes them from the full “detection–forensics–knowledge” workflow engine designed in this study.

### 2.2.2 Applications of Large Models and RAG in Cybersecurity

Large models and RAG have been applied in threat intelligence, vulnerability understanding, and alert interpretation. Cheng et al. [14] proposed CTINexus, which uses large language models to automatically construct network threat intelligence knowledge graphs, converting unstructured intelligence into structured knowledge graphs.

Sun et al. [15] proposed LLM4Vuln, which disentangles and evaluates the vulnerability reasoning ability of large models and applies it to zero-day vulnerability detection. Vinod Krishna B [16] introduced the AttackQA dataset to support cybersecurity operation assistance via fine-tuned open-source large models.

Ismail et al. [17] adopted hyper-automation and agentic large models to enhance investigation quality and SOC efficiency for security orchestration and automated response (SOAR). Tariq et al. [18] systematically reviewed alert fatigue in security operations centers, summarizing research challenges and opportunities.

Overall, existing studies have advanced single-point capabilities using large models and RAG, but remain insufficient in full workflow orchestration and LLM-RAG-driven adaptive processes. This study fills this gap with an adaptive dynamic workflow engine.

# 2 相关工作

网络攻击告警研判从人工辅助走向智能化与自动化，学界普遍关注检测效率、准确性与对新型威胁的适应性。本节按传统告警分析方法与智能告警分析技术两类，结合参考文献的实际写作内容进行综述。

## 2.1 传统网络攻击告警分析方法研究

### 2.1.1 人工分析辅助与主机/异常检测

人工研判面临信息过载与经验依赖。在主机与异常检测方面，Bejtlich [6] 在《The Tao of Network Security Monitoring: Beyond Intrusion Detection》中提出NSM作战框架，但实践中误报率仍较高。Hassan 等 [7] 提出 NoDoze，利用因果依赖图对告警进行自动化溯源分诊：基于历史频率异常评分，并通过图上网络扩散算法传播评分，生成优先化的子图供分析；在 364 条威胁告警上的评估显示误报量减少约 86%，但方法依赖因果依赖图的构建，在不同 SOC 环境中的可迁移性仍受限于依赖图质量与数据可得性。

### 2.1.2 固化检测引擎与 SOAR 系统

传统 IDS 与固化引擎存在明显局限。Sommer 与 Paxson [8] 指出，基于“封闭世界”假设的传统网络入侵检测难以有效应对未知攻击，并讨论了将机器学习用于网络 IDS 的机遇与挑战。在异常检测算法层面，Liu 等 [9] 提出 Isolation Forest，通过随机子采样与隔离树实现高效异常检测，被广泛用于单点异常识别，但其在复杂攻击场景下的泛化与可解释性仍有限。Bridges 等 [10] 针对六款商用 SOAR 工具开展了首次大规模实证用户研究，在构建的网络靶场环境中，通过 24 名安全运营人员的实际操作验证发现：使用 SOAR 工具可显著提升安全事件调查效率、减少分析师上下文切换，但会导致工单记录的准确率与完整性有所下降；SOAR 的实际效果高度依赖工具配置，过度自动化反而会影响分析质量。该工作从实证角度验证了 SOAR 在提升运营效率的同时存在易用性、自动化程度与任务质量之间的平衡问题，为本研究面向实际安全运营场景设计自适应响应机制提供了现实依据。

## 2.2 智能告警分析技术研究现状

### 2.2.1 机器学习驱动的误报过滤与威胁分类

机器学习被广泛用于告警分析中的误报过滤与威胁分类。Bhati 等 [11] 采用基于 XGBoost 的集成方法进行网络入侵检测，实现更优的 “偏差-方差” 权衡，提升了集成式 IDS 的效率和准确率。Hou 等 [12] 提出层次化长短期记忆网络（LSTM）用于网络流量上的网络攻击检测，利用序列建模提升对攻击模式的刻画能力。Huang 等 [13] 结合图神经网络（GNN）与跨协议分析检测恶意 IP，在恶意 IP 识别任务上取得较高准确率。这些工作主要针对检测或分类单环节进行优化，缺乏对告警全流程工作流的编排与调度，与本研究面向“检测—取证—知识”全流程的引擎设计形成区别。

### 2.2.2 大模型与 RAG 在安全领域的应用

大模型与 RAG 在威胁情报、漏洞理解与告警解读等方面已有应用。Cheng 等 [14] 提出 CTINexus，利用大语言模型自动构建网络威胁情报知识图谱，实现从非结构化情报到结构化知识图的生成。Sun 等 [15] 提出 LLM4Vuln，对大模型的漏洞推理能力进行解耦与评估，并应用于零日漏洞识别等场景。Vinod Krishna B [16] 提出 AttackQA 数据集，用于支持基于微调与开源大模型的网络安全运维辅助任务。Ismail 等 [17] 面向 SOC 中的安全编排与自动响应（SOAR），采用超自动化与智能体式大模型（Agentic AI）提升调查质量与 SOC 效率。Tariq 等 [18] 对安全运营中心中的告警疲劳问题进行了系统综述，归纳了研究挑战与机遇。综合来看，现有研究在利用大模型与 RAG 增强单点能力方面已有进展，但在全流程工作流编排与 LLM-RAG 驱动的流程自适应方面仍显不足；本研究通过自适应动态工作流引擎填补该空白。

**3** **Alert Analysis Process**

In modern enterprise SOC operations, analysts must handle massive alerts, but traditional experience/rule-based judgment causes false alarm backlogs, fatigue, and delays in high-volume/complex scenarios.

Traditional analysis starts with multi-source telemetry aggregation: Syslog, EDR (e.g., CrowdStrike), Zeek, CloudTrail, SPAN/TAP collect data, stored in Splunk via Logstash for structured output. Data is standardized (key field extraction, deduplication [20]), then prioritized via rules/asset assessment/machine learning for L1 screening or L2 deep analysis [21]. L1 uses EDR/IP reputation for status marking; L2 builds timelines with MITRE ATT&CK threat profiling [22]

传统分析从多源遥测汇聚开始：Syslog、EDR（如 CrowdStrike）、Zeek、CloudTrail、SPAN/TAP 等采集数据，经 Logstash 存入 Splunk 形成结构化输出。数据经标准化（关键字段抽取、去重 [20]）后，通过规则/资产评估/机器学习进行 L1 筛选或 L2 深度分析 [21]。L1 利用 EDR/IP 信誉进行状态标记，L2 结合 MITRE ATT&CK 威胁画像构建时间线 [22]。

The authoritative guide to incident response [19] and MITRE ATT&CK attack modeling framework [22] issued by NIST provide the core theoretical and practical basis for the optimization of various network security operation processes, and the landing of intelligent alarm classification and triage system [23] is the concrete application of such process optimization in SOC scenarios.

NIST 发布的事件响应权威指南 [19] 与 MITRE ATT&CK 攻击建模框架 [22]，为各类网络安全运营流程的优化提供了核心理论与实践依据，而智能化告警分类分诊系统的落地 [23]，正是此类流程优化在 SOC 场景中的具体应用体现。

**4** **MAIN METHOD**

Traditional network security alarm analysis faces issues: manual analysis has massive redundancy, fragmented data, and high knowledge dependence; fixed engines lack unknown threat generalization, context, and timely knowledge updates. This study uses LLM for multi-source data parsing, integrates RAG with a vector knowledge base to fill LLM gaps, forming a full-chain intelligent solution.

**4****.1** **Overview of Model Architecture**

Centered on "layered detection, clue enhancement, knowledge empowerment", the framework is supported by three core components. Following the "classification → processing → collaboration → integrated output" workflow, it forms a closed loop with LLM to address accuracy, context, and knowledge lag issues in traditional judgment.

As shown in Figure 2, the framework initiates with "problem input" and has four stages for full-chain intelligence. Three input types—unstructured traffic files, structured data, natural language conversations—are classified, verified, and parsed into respective branches: Scapy-parsed PCAP files feed threat detection agents; structured data clues go to forensics agents; conversation keywords are retrieved from the vector knowledge base. With dual-layer detection (three integrated engines), internal-external intelligence association, and semantic retrieval, LLM integrates multi-source data to output standardized reports.

**Figure 2: Flowchart of the Overall Framework for the Cybersecurity Alert Analysis Large Model**

**4****.****2** **Threat Detection Agent**

The threat detection agent is the front-end core of the SecAgent framework, responsible for real-time threat identification and classification of multi-source heterogeneous network traffic, aiming to solve the problems of independent alarm rules, difficulty in identifying new threats, and lack of attack chain characterization in existing SOC security devices.

As shown in Figure 3, this intelligent agent integrates Snort rule engine (covering known threats), custom rule engine (adapted to internal networks), and machine learning engine (capturing unknown anomalies) to form automated secondary analysis capabilities; Realize the transformation from "alarm stacking" to "precise labeling" and reduce manual workload.

**Figure 3: Threat detection intelligent agent framework flowchart**

*4**.**2**.1* *SNORT RULE MATCHING ENGINE*

This module is based on the Snort open-source rule library and custom rules, implementing millisecond level matching of known threats such as SQL injection. As shown in Figure 4, through the process of "rule construction traffic parsing indexing matching output", output JSON results including timestamps to provide standardized input for subsequent modules.

**Figure 4: Snort rule matching engine flowchart**

The rule library includes 4,000+ open-source Emerging Threats rules (e.g., SQL injection) and enterprise-specific custom rules (sid ≥ 1000000, e.g., SYN Flood), all adhering to Snort syntax. Scapy-parsed raw traffic undergoes multi-layer detection, outputting JSON results (timestamps, threat classification) for subsequent modules.

*4**.**2**.**2* *CUSTOM RULE DETECTION ENGINE*

As a traditional feature-based intrusion detection system, Snort's core limitations have been confirmed by multiple studies: firstly, its single-threaded architecture leads to insufficient processing capacity for high-speed network traffic (e.g., 10 Gbps) and a higher packet loss rate compared with Suricata [24]; Secondly, it has limited ability to utilize multi-core hardware resources, which makes it difficult to adapt to the increasing volume of network packets caused by digital convergence [24]; Thirdly, its default rule set tends to trigger a high false positive rate, and its alarm aggregation capability is insufficient without additional optimization [26].

作为传统的基于特征的入侵检测系统，Snort 的核心局限性已被多项研究证实：首先，其单线程架构导致处理能力不足以应对高速网络流量（例如 10 Gbps），且丢包率高于 Suricata [24];其次，它在利用多核硬件资源方面能力有限，这使得适应数字融合带来的日益增长的网络数据包量变得困难[24];第三，其默认规则集往往会触发较高的误报率，且其报警聚合能力在没有额外优化的情况下难以实现[26]。

*4**.**2**.**3* *MACHINE LEARNING DETECTION ENGINE*

For threat types that are inherently subtle and difficult to visually detect through conventional means, the intelligent engine fully leverages advanced machine learning self-learning capabilities to deeply mine and extract intricate attack features from three widely recognized and publicly available benchmark datasets: KDD Cup 99 [30], CIC-IDS2017 [31], and UNSW-NB15 [32]. By systematically analyzing the rich and diverse attack scenarios, traffic patterns, and behavioral characteristics encapsulated within these datasets, the engine constructs a robust, multi-dimensional threat recognition mechanism that integrates feature engineering, pattern matching, and adaptive learning.

*（中文）* 针对本身隐蔽、难以通过常规手段直观识别的威胁类型，智能引擎充分借助机器学习自学习能力，从三个广泛认可的公开基准数据集 KDD Cup 99 [30]、CIC-IDS2017 [31] 与 UNSW-NB15 [32] 中深度挖掘并提取细粒度攻击特征；通过对这些数据集中蕴含的多样化攻击场景、流量模式与行为特征进行系统分析，构建融合特征工程、模式匹配与自适应学习的多维威胁识别机制。

**Figure 6: Flow Chart of Machine Learning Detection Engine**

The three major datasets cover the main types of network attacks, as the following table 1 shows:

TABLE 1: MAIN NETWORK ATTACK TYPES COVERED BY THE THREE MAJOR DATASETS

| Dataset     | Main Attack Types                                            |
| ----------- | ------------------------------------------------------------ |
| KDD99       | Denial of Service (DoS), Remote to Local (R2L), User to Root (U2R), Probing |
| CIC-IDS2017 | Brute-force attack, Heartbleed vulnerability attack, SQL injection, DDoS attack, Botnet |
| UNSW-NB15   | Backdoor attack, Trojan attack, Phishing attack, Malicious code attack, File infiltration attack |

As shown in Figure 6, the CICFLOW-style feature set is used as input, with models including Isolation Forest [33], XGBoost [34], LightGBM [35], and Random Forest. The three datasets are preprocessed (sample balancing, feature normalization) to build a multi-dataset multi-model detection library. During detection, Isolation Forest screens normal traffic, while XGBoost and other models classify features by weighted confidence; pre-trained models are invoked for inference, and new models can be added to enhance performance.

*（中文）* 如图 6 所示，以 CICFLOW 风格特征集为输入，模型包括 Isolation Forest [33]、XGBoost [34]、LightGBM [35] 与随机森林。对三类数据集进行预处理（样本均衡、特征归一化）以构建多数据集多模型检测库。检测时由 Isolation Forest 筛除正常流量，XGBoost 等模型按加权置信度对特征分类；调用预训练模型进行推理，并可扩展新模型以提升性能。

**4****.****3** **Clue Forensics Agent**

The clue forensics agent is a "threat context enhancement layer" that takes detection results as input, confirms the authenticity and impact range of threats through threat intelligence matching, asset impact quantification, and diffusion path deduction, and supports decision-making.

It connects external intelligence such as MITRE ATT&CK [22] with internal case verification threats, evaluates the impact based on asset tables, and generates propagation maps according to network topology. At the same time, a dual source database is established, and after Sentence-BERT [36] encoding and FAISS retrieval, standardized reports containing target assets are output to provide high-quality input for LLM.

*（中文）* 将 MITRE ATT&CK [22] 等外部情报与内部案件核验威胁关联，结合资产表评估影响并依网络拓扑生成扩散图。同时建立双源数据库，经 Sentence-BERT [36] 编码与 FAISS 检索后，输出包含目标资产的标准化报告，为 LLM 提供高质量输入。

**4****.****4** **Knowledge Base: Multi-Dimensional Structure, Authoritative Sources, and Collaborative Retrieval Mechanism**

As the core of RAG, the vector knowledge base provides real-time authoritative security context for LLMs, addressing their knowledge lag and hallucination. It embeds structured/unstructured data for efficient semantic retrieval, with threat labels as vector retrieval keys to ensure accurate, timely, and credible knowledge input.

The overall authoritative data sources of the knowledge base are shown in Table 2.

The knowledge base retrieval agent efficiently obtains knowledge through knowledge vectorization, semantic retrieval, and RAG context generation, in collaboration with workflow engines. When importing knowledge base text, it is embedded into a model and converted into high-dimensional vectors, which are stored in a vector database. After obtaining threat labels, the workflow engine vectorizes the labels and alarm contexts, performs semantic similarity retrieval to improve recall and accuracy, integrates Top-K knowledge blocks and structures them as external knowledge inputs for LLM, and enhances the accuracy and credibility of research conclusions.

 

 



TABLE 2 STATISTICAL TABLE OF KNOWLEDGE BASE DATA SOURCES

| Knowledge Category    | Subcategory                                                 | Authoritative Data Sources & Platforms                       | Core Function                                                |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Threat Intelligence   | Vulnerability Knowledge                                     | CNNVD, CWE                                                   | Provides national-level vulnerability details and common defect types to assist in assessing asset vulnerability. |
| IoC Intelligence      | ThreatBook and other commercial threat intelligence sources | Offers real-time, granular intrusion indicators (e.g., malicious IPs, URLs, domains) for clue forensics and threat matching. |                                                              |
| Attack Technology     | ATT&CK Knowledge Base                                       | MITRE ATT&CK                                                 | Provides standardized attack chain positioning and context interpretation for alerts, ensuring industry uniformity in technical descriptions. |
| CAPEC Attack Patterns | CAPEC                                                       | Supplements descriptions of abstract attack patterns and methods to enhance understanding of attack paths and intentions. |                                                              |
| Defense Technology    | D3FEND Knowledge Base                                       | MITRE D3FEND                                                 | Maps actionable defense technologies to identified attack techniques, guiding defense strategy formulation. |
| Incident Response     | Disposal Processes & Specifications                         | NIST SP 800-61 Rev.2, SANS, Information Security Emergency Response Plan Specifications | Guides subsequent standardized disposal processes and emergency management based on international standards and industry best practices. |
| Compliance Standards  | National Standards                                          | Based on [openstd.samr.gov.cn](https://openstd.samr.gov.cn/) | Ensures that analysis and disposal processes meet national and industry security compliance requirements. |
| Comprehensive Reports | Analysis Reports                                            | Security incident reports and annual reports from various security vendors | Provides historical cases and trend data for LLMs to conduct inductive reasoning, enhancing their ability to assess incident impacts and long-term risks. |

TABLE 3: ABLATION EXPERIMENT RESULTS OF THREAT DETECTION ENGINE WITH DIFFERENT METHOD COMBINATIONS (F1-SCORE, UNIT: %)

| Experimental Setup            | CIC-IDS2017 | UNSW-NB15   | KDD99       | Average F1-Score |
| ----------------------------- | ----------- | ----------- | ----------- | ---------------- |
| A:All components              | 98.3        | 95.8        | 99.3        | 97.8             |
| B:Only Snort+ML model         | 97.8 (-0.5) | 94.1 (-1.7) | 99.0 (-0.3) | 96.9 (-0.9)      |
| C:Only custom rules+ML models | 96.5 (-1.8) | 92.7 (-3.1) | 98.5 (-0.8) | 95.9 (-1.9)      |

 



**5** **Experimental Analysis**

**5****.****1** **Experimental Design**

This study aims to improve security personnel’s efficiency in handling massive alerts. To this end, the model must excel in threat detection accuracy, usability, and overall operational efficiency.

5.1.1 EXPERIMENT ON THE ACCURACY OF THE THREAT DETECTION ENGINE

This experiment is designed to verify the basic detection capability of the integrated detection engine. It evaluates the engine's capability using accuracy, precision, recall, and F1 Score based on the labeled public datasets CIC-IDS2017, UNSW-NB15, and KDD99. In addition, to demonstrate the superiority of the proposed model, relevant studies in recent years are selected for comparison (see references below).

①Comparative Study 1[27]: Reinforcement Learning Transfer Framework

*（中文）* ① 对比研究 1 [27]：强化学习迁移框架

②Comparative Study 2[28]: Hybrid IDS Based on GCN and Transformer (GCN-2-Former)

*（中文）* ② 对比研究 2 [28]：基于 GCN 与 Transformer 的混合 IDS（GCN-2-Former）

③Comparative Study 3[29]: Deep Learning Based Intrusion Detection

*（中文）* ③ 对比研究 3 [29]：基于深度学习的入侵检测

5.1.2 EXPERIMENT ON THE USABILITY OF THE THREAT DETECTION ENGINE

This experiment verifies the threat detection engine’s throughput and latency for massive network data, ensuring real/near-real-time production adaptability. Tcpreplay replays high-rate background traffic with injected attack packets, evaluated via throughput, latency, and resource utilization.

5.1.3 EXPERIMENT ON THE OVERALL USABILITY OF THE MODEL

This experiment is conducted to verify the usability of the model proposed in this study in improving the analysis efficiency of security analysts, covering two aspects: analysis accuracy and analysis speed. Fifty independent traffic files are randomly selected from the dataset to form a test set. First, a review team composed of 2-3 security analysts conducts independent manual analysis on each alert in the test set and provides the analysis results. Second, the alerts in the test set are input into the LLM workflow to obtain the output results of the LLM.

The experiment focuses on the accuracy of threat detection, operational efficiency, and knowledge retrieval usability of the integrated model. The specific design is as follows:

Threat detection accuracy experiment: Using accuracy, precision, recall, and F1 Score as indicators, validate the performance of the integrated engine through benchmark testing, ablation experiments, and comparative experiments.

Threat detection efficiency experiment: Using throughput, latency, and resource utilization as indicators, monitoring system performance under different loads through replay traffic to verify real-time performance.

Model overall usability experiment: Select 50 traffic files, compare the accuracy and analysis efficiency of LLM and manual judgment, and have a third-party blind review report quality.

**5****.****2** **Data Introduction**

The experimental data is divided into three categories, supporting different experimental objectives:

Benchmark dataset: used for accuracy verification, including CIC-IDS2017, UNSW-NB15, KDD99.

Simulated dataset: used for performance testing, based on CIC-IDS2017 normal traffic, IP simulated internal network segment modified by tcprewrite, tcpreplay replayed at 1Gbps/5Gbps/10Gbps, injecting 1% -5% attack traffic.

Knowledge base data: supports LLM retrieval, integrates Snort rule base, ATT&CK MITRE framework, CVE vulnerability summary, and security vendor reports.

**5****.****3** **Accuracy Experiment**

As shown in Figure 7-10, this is the detection performance of the threat detection engine proposed in this study on three different datasets. The three benchmark datasets were each randomly divided into training set, validation set, and test set at a ratio of 70% : 15% : 15%. The integrated detection engine was run on the test set, and the accuracy, precision, recall, and F1-Score were calculated. It can be seen that the threat detection engine proposed in this study exhibits good detection capability across all three datasets. Additionally, based on its performance on different datasets, the engine achieves the best results on the KDD99 dataset, while its performance on the UNSW-NB15 dataset is relatively lower. This is because the attack patterns in KDD99 are relatively simple with distinct feature differentiation, whereas UNSW-NB15 includes more diverse attack types, more complex network environment simulations, and traffic features that are closer to those in the real world

Ablation experiments (Table 3) verify the necessity of integrating multiple detection methods. Scheme A (all components) achieves an average F1-score of 97.8%—optimal across datasets. Scheme B (Snort+ML) sees a 0.9% average drop (1.7% on UNSW-NB15), while Scheme C (custom rules+ML) declines 1.9% (3.1% on UNSW-NB15). This confirms multi-method integration enhances robustness via complementary advantages.

**Figure 7: Performance Metrics Comparison Across**

**Figure 8: ROC Curves for Threat Detection Engine**

**Figure 9: Performance Trends Across Datasets**

 



 

**Figure 10: Confusion Matrix of CIC-IDS2017/UNSW-NB15/KDD99**

**Figure 11: Accuracy and F1-Score Comparison Across Methods and Datasets**

 



To verify the engine’s superiority, comparisons with recent studies (Figure 11) show it outperforms all baselines across datasets. vs. the best method (Transformer+ANFIS): CIC-IDS2017 (1.3% accuracy/F1 gain), UNSW-NB15 (1.8% accuracy, 2.2% F1 gain), KDD99 (0.6% accuracy, 0.8% F1 gain). This advantage comes from innovative feature extraction, anomaly detection, and classification design, ensuring strong generalization.

**5****.****4** **Performance Experiment**

TABLE 4: PERFORMANCE METRICS (THROUGHPUT) AND RESOURCE CONSUMPTION (CPU, MEMORY) OF THREAT DETECTION ENGINE UNDER DIFFERENT TRAFFIC LOADS

| Traffic load (Gbps) | Average throughput(Mpps) | Throughput (Gbps) | CPU usage  (%) | Memory usage (GB) |
| ------------------- | ------------------------ | ----------------- | -------------- | ----------------- |
| 1                   | 1.49                     | 0.995             | ~15%           | 2.1               |
| 5                   | 7.46                     | 4.98              | ~65%           | 2.5               |
| 10                  | 9.87                     | 6.58              | ~95%           | 2.8               |

As shown in Table 4, the engine maintains near-line-speed processing (7.46 Mpps/4.98Gbps) from 1Gbps to 5Gbps, with good scalability. At 10Gbps, throughput drops to 9.87 Mpps/6.58Gbps due to bottlenecks. CPU utilization rises sharply from ~15% to ~95% (memory stable at 2.1-2.8GB). The engine performs stably below 5Gbps; CPU is the main bottleneck near 10Gbps, requiring high-load optimization or distributed deployment.

**5****.****5** **Usability Experiment**

The model’s usability for alert analysis is evaluated via attack type accuracy, efficiency, and report usability (Table 5). On 50 traffic files, human analysts (92% accuracy) outperform the LLM (86%), with humans excelling in complex threats. LLM misjudgments focus on encrypted C&C and fine-grained attack classification; human errors stem from over-sensitivity to legitimate traffic and poor new variant identification. Still, the LLM’s 86% accuracy effectively assists security analysts.

TABLE 5: COMPARISON OF ATTACK TYPE JUDGMENT RESULTS BETWEEN THE PROPOSED MODEL (LLM) AND MANUAL ANALYSTS ON 50 TRAFFIC FILES

| Object of Evaluation | Correct Sample Size | Accuracy | Misjudgment Cases                                            |
| -------------------- | ------------------- | -------- | ------------------------------------------------------------ |
| LLM                  | 43                  | 86%      | Sample # 12: Encryption C&C heartbeat misclassified as Normal; Sample # 37: SQL injection misclassified as path traversal |
| Manual analysis      | 46                  | 92%      | Sample # 08: Legitimate CDN traffic misclassified as DDoS; Sample # 29: Unrecognized Brute Force variant |

While slightly less accurate than humans, the LLM achieves a qualitative speed improvement (Table 6). It analyzes all 50 samples in 8 minutes (≈10.2s per sample) vs. 6h23m for humans (≈7.7min per sample)—45x higher efficiency. This gap underscores its automation potential, enabling near-real-time threat analysis for large-scale rapid-response security operations.

TABLE 6: COMPARISON OF THREAT ASSESSMENT SPEED (TIME CONSUMPTION PER SAMPLE) BETWEEN THE PROPOSED MODEL (LLM) AND MANUAL ANALYSTS

| Object of Evaluation | Average Time Consumption/Sample Size | Efficiency multiplier (LLM compared to manual) |
| -------------------- | ------------------------------------ | ---------------------------------------------- |
| LLM                  | ~10.2 seconds                        | ~45  times                                     |
| Manual analysis      | ~7.7 minutes                         | 1x  (baseline)                                 |

Beyond attack type accuracy and analysis speed, the final analysis report is also critical. Experts blind-reviewed and scored 100 reports from the LLM and human team (Table 7).

TABLE 7: BLIND REVIEW SCORING RESULTS OF THREAT ASSESSMENT REPORTS GENERATED BY THE PROPOSED MODEL (LLM) AND MANUAL ANALYSIS TEAMS (TOTAL 100 REPORTS)

| Object of Evaluation | Accuracy | Completeness | Clarity | Operability | Comprehensive Average Value |
| -------------------- | -------- | ------------ | ------- | ----------- | --------------------------- |
| LLM                  | 4.0      | 4.6          | 4.7     | 3.8         | 4.28                        |
| Manual Analysis      | 4.8      | 4.4          | 4.0     | 4.6         | 4.45                        |

An in-depth analysis was conducted on the scoring results for each dimension, as shown in Table 8.



TABLE 8: IN-DEPTH ANALYSIS OF BLIND REVIEW SCORES BY DIMENSION (LLM VS. MANUAL ANALYSIS)

| Scoring Dimension        | LLM                                                          | Manual Analysis                                              |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Accuracy(4.0 vs 4.8)     | Conclusions are generally reliable, but its reasoning ability is inferior to the human expert team in the most complex cases that require out-of-pattern thinking. | Team discussions can effectively correct personal biases and misjudgments, achieving near-expert-level accuracy. |
| Completeness(4.6 vs 4.4) | Strength. It strictly follows templates, barely omitting any preset analysis dimensions, with stable output. | Occasionally overlooks the elaboration of "potential impacts" or "other possibilities" due to focus on a single attack point. |
| Clarity(4.7 vs 4.0)      | Significant strength. Reports have a unified structure and fluent, standardized language, greatly improving reading and comprehension efficiency. | Reports integrate ideas from multiple people, sometimes appearing lengthy or with inconsistent terminology. |
| Operability(3.8 vs 4.6)  | Main weakness. Recommendations tend to be general and conservative (e.g., "block IPs," "apply patches"), lacking refined response steps tailored to specific IT environments. | Strength. It can align with the organization’s actual security policies and infrastructure to provide specific, feasible suggestions (e.g., "update specific policies on firewalls," "contact a system administrator to check logs"). |

 



**6** **CONCLUSION**

This study proposes a RAG-enhanced generative large model-driven dynamic workflow engine framework for SOC alert analysis, addressing low manual efficiency and fixed process engine inflexibility to break through from "fixed process execution" to "intelligent process orchestration". Core contributions include a RAG-LLM-centric dynamic workflow engine overcoming traditional SOAR rigidity, a multi-agent collaborative mechanism forming a "detection-forensics-knowledge integration" closed loop, and valid experimental results—95.8%-99.3% F1-Score across three datasets, 6.58Gbps throughput at 10Gbps traffic, 45x manual efficiency, and ≤1.36% false alarm rate. The framework balances accuracy and real-time performance; future work will optimize model robustness, build knowledge update mechanisms, and expand cross-domain collaboration capabilities.

**ACKNOWLEDGMENTS**

I would like to express my sincere gratitude to all those who have contributed to this study with their support and assistance.

**REFERENCES**

[1] Verizon. 2024. 2024 Data Breach Investigation Report. Technical Report. Verizon.Retrieved from https://www.verizon.com/business/resources/reports/2024-dbir-data-breach-investigations-report.pdf

[2] Weiwei Mo. 2024. Analysis of National Cybersecurity Risk Situation in 2024. China Information Security. Retrieved from https://www.secrss.com/articles/77661

[3] Daily Security Review. 2024. Halliburton ransomware attack costs energy giant $35 million. Daily Security Review (Nov. 12, 2024). Retrieved from https://dailysecurityreview.com/security-spotlight/halliburton-ransomware-attack-costs-energy-giant-35-million/

[4] Enoch Agyepong, Yulia Cherdantseva, Philipp Reinecke, and Pete Burnap. 2020. Challenges and performance metrics for security operations center analysts: A systematic review. Journal of Cyber Security Technology 4, 2 (2020), 125–152. DOI:https://doi.org/10.1080/23742917.2019.1698178

[5] Ansam Khraisat, Iqbal Gondal, Peter Vamplew, and Joarder Kamruzzaman. 2019. Survey of intrusion detection systems: techniques, datasets and challenges. Cybersecurity 2, 20 (2019). DOI:https://doi.org/10.1186/s42400-019-0038-7

[6] Richard Bejtlich. 2004. The Tao of Network Security Monitoring: Beyond Intrusion Detection. Pearson Education.DOI:https://dl.acm.org/doi/book/10.5555/1024187

[7] Wajih Ul Hassan, Shengjian Guo, Ding Li, Zhengzhang Chen, Kangkook Jee, Zhichun Li, and Adam Bates. 2019. NoDoze: Combatting threat alert fatigue with automated provenance triage. In Proceedings of the 26th Network and Distributed System Security Symposium (NDSS 2019). DOI:https://doi.org/10.14722/ndss.2019.23349

[8] Robin Sommer and Vern Paxson. 2010. Outside the closed world: On using machine learning for network intrusion detection. In Proceedings of the 2010 IEEE Symposium on Security and Privacy (S&P 2010). IEEE, 305–316. DOI: https://doi.org/10.1109/SP.2010.25

[9] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. 2008. Isolation Forest. In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM). IEEE. DOI:https://doi.org/10.1109/ICDM.2008.17

[10] Robert A. Bridges, Ashley E. Rice, Sean Oesch, Jeffrey A. Nichols, Cory Watson, Kevin Spakes, Savannah Norem, Mike Huettel, Brian Jewell, Brian Weber, Connor Gannon, Olivia Bizovi, Samuel C. Hollifield, and Samantha Erwin. 2023. Testing SOAR tools in use. Computers & Security 126 (2023), 103201. DOI:https://doi.org/10.1016/j.cose.2023.103201

[11] Bhoopesh Singh Bhati, Garvit Chugh, Fadi Al-Turjman, and Nitesh Singh Bhati. 2021. An improved ensemble based intrusion detection technique using XGBoost. Transactions on Emerging Telecommunications Technologies 32, 6 (2021), e4076. DOI:https://doi.org/10.1002/ett.4076

[12] Haixia Hou, Yingying Xu, Menghan Chen, Zhi Liu, Wei Guo, Mingcheng Gao, Yang Xin, and Lizhen Cui. 2020. Hierarchical Long Short-Term Memory Network for Cyberattack Detection. IEEE Access 8 (2020), 55611–55622. DOI:https://doi.org/10.1109/ACCESS.2020.2983953

[13] Yonghong Huang, Joanna Negrete, John Wagener, Celeste Fralick, Armando Rodriguez, Eric Peterson, and Adam Wosotowsky. 2022. Graph neural networks and cross-protocol analysis for detecting malicious IP addresses. Complex & Intelligent Systems 9 (2022), 1977–1992. DOI:https://doi.org/10.1007/s40747-022-00838-y

[14] Yutong Cheng, Osama Bajaber, Saimon Amanuel Tsegai, Dawn Song, and Peng Gao. 2024. CTINexus: Automatic Cyber Threat Intelligence Knowledge Graph Construction Using Large Language Models. arXiv:2410.21060. DOI: https://doi.org/10.48550/arXiv.2410.21060

[15] Yuqiang Sun, Daoyuan Wu, Yue Xue, Han Liu, Wei Ma, Lyuye Zhang, Yang Liu, and Yingjiu Li. 2024. LLM4Vuln: A unified evaluation framework for decoupling and enhancing LLMs’ vulnerability reasoning. arXiv:2401.16185. DOI: https://doi.org/10.48550/arXiv.2401.16185

[16] Vinod Krishna B. 2024. AttackQA: Development and adoption of a dataset for assisting cybersecurity operations using fine-tuned and open-source LLMs. arXiv:2411.01073. DOI: https://doi.org/10.48550/arXiv.2411.01073

[17] Ismail, Rahmat Kurnia, Zilmas Arjuna Brata, Ghitha Afina Nelistiani, Shinwook Heo, Hyeongon Kim, and Howon Kim. 2025. Toward Robust Security Orchestration and Automated Response in Security Operations Centers with a Hyper-Automation Approach Using Agentic Artificial Intelligence. Information 16, 5 (2025), 365. DOI:https://doi.org/10.3390/info16050365

[18] Shahroz Tariq, Mohan Baruwal Chhetri, Surya Nepal, and Cecile Paris. 2025. Alert Fatigue in Security Operations Centres: Research Challenges and Opportunities. ACM Computing Surveys 57, 9 (2025), Article 224. DOI:https://doi.org/10.1145/3723158

[19] Alex Nelson, Sanjay Rekhi, Murugiah Souppaya, Karen Scarfone. Incident Response Recommendations and Considerations (SP 800-61 Rev. 3). National Institute of Standards and Technology. 2025. NIST Special Publication 800-61 Rev. 3. DOI: https://doi.org/10.6028/NIST.SP.800-61r3

[20] Fatemeh Jalalvand, Mohan Baruwal Chhetri, Surya Nepal, and Cecile Paris. 2024. Alert prioritisation in security operations centres: A systematic survey on criteria and methods. ACM Computing Surveys 57, 2 (2024), Article 44. DOI: https://doi.org/10.1145/3695462

[21] Shahroz Tariq, Mohan Baruwal Chhetri, Surya Nepal, and Cecile Paris. 2025. Alert Fatigue in Security Operations Centres: Research Challenges and Opportunities. ACM Computing Surveys 57, 9 (2025), Article 224. DOI:https://doi.org/10.1145/3723158

[22] Blake E. Strom, Andy Applebaum, Doug P. Miller, Kathryn C. Nickels, Adam G. Pennington, and Cody B. Thomas. 2018. MITRE ATT&CK: Design and philosophy. MITRE Technical Report. Retrieved from https://www.mitre.org/sites/default/files/2021-11/prs-19-01075-28-mitre-attack-design-and-philosophy.pdf

[23] Melissa Turcotte, François Labrèche, and Serge-Olivier Paquette. 2025. Automated Alert Classification and Triage (AACT): An Intelligent System for the Prioritisation of Cybersecurity Alerts. arXiv:2505.09843. DOI: https://doi.org/10.48550/arXiv.2505.09843

[24] Wonhyung Park and Seongjin Ahn. 2017. Performance comparison and detection analysis in Snort and Suricata environment. Wireless Personal Communications 94 (2017), 241–252. DOI: https://doi.org/10.1007/s11277-016-3209-9

[25] Jason S. White, Thomas Fitzsimmons, and Jeanna N. Matthews. 2013. Quantitative analysis of intrusion detection systems: Snort and Suricata. In Cyber Sensing 2013, Vol. 8757. SPIE, 10–21. DOI:https://doi.org/10.1117/12.2015616

[26] Syed A. R. Shah and Biju Issac. 2018. Performance comparison of intrusion detection systems and application of machine learning to Snort system. Future Generation Computer Systems 80 (2018), 157–170. DOI:https://doi.org/10.1016/j.future.2017.10.016

[27] Mingshu He, Xiaojuan Wang, Peng Wei, Liu Yang, Yinglei Teng, and Renjian Lyu. 2024. Reinforcement learning meets network intrusion detection: A transferable and adaptable framework for anomaly behavior identification. IEEE Transactions on Network and Service Management 21, 2 (2024), 2477–2492. DOI:https://doi.org/10.1109/TNSM.2024.3352586




[28] Jingkang Zhang, X. Fan, Z. Zhao. 2025. A hybrid intrusion detection model based on dynamic spatial-temporal graph neural network in in-vehicle networks. Scientific Reports 15, 34736 (2025). https://doi.org/10.1038/s41598-025-18401-3

[29] R. Vinayakumar, Mamoun Alazab, K. P. Soman, Prabaharan Poornachandran, Ameer Al-Nemrat, and S. Venkatraman. 2019. Deep learning approach for intelligent intrusion detection system. IEEE Access 7 (2019), 41525–41550. DOI: https://doi.org/10.1109/ACCESS.2019.2895334

[30] Mahbod Tavallaee, Ebrahim Bagheri, Wei Lu, and Ali A. Ghorbani. 2009. A detailed analysis of the KDD CUP 99 data set. In Proceedings of the 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA 2009). IEEE, 1–6. DOI: https://doi.org/10.1109/CISDA.2009.5356528

[31] Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani. 2018. Toward generating a new intrusion detection dataset and intrusion traffic characterization. In Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP). SCITEPRESS, 108–116. DOI:https://doi.org/10.5220/0006639801080116

[32] Nour Moustafa and Jill Slay. 2015. UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). In 2015 Military Communications and Information Systems Conference (MilCIS). IEEE, 1–6. DOI:https://doi.org/10.1109/MilCIS.2015.7348942

[33] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. 2008. Isolation Forest. In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM). IEEE, 413–422. DOI:https://doi.org/10.1109/ICDM.2008.17

[34] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). ACM, 785–794. DOI:https://doi.org/10.1145/2939672.2939785

[35] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. 2017. LightGBM: A highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems 30 (NIPS 2017). Curran Associates, Inc., 3146–3154. DOI: https://dl.acm.org/doi/10.5555/3294996.3295074

[36] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, 3982–3992. DOI:https://doi.org/10.18653/v1/D19-1410
