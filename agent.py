import os
import json
import time
import logging
import re
import hashlib
from typing import Dict, List, Any
from knowledge_retriever import DocumentKnowledgeRetriever

from engine.ids_analyzer import load_rules, RuleEngine, analyze_pcap as snort_analyze_pcap
from engine.rule_engine import EnhancedRuleEngine, PcapAnalyzer as RuleEnginePcapAnalyzer
from engine.ml_engine import AnomalyDetector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("alert_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AlertAgent')


class DeepSeekR1Service:
    """DeepSeek-R1 LLM服务封装（优化版）"""

    def __init__(self, model_path="llm_models/deepseek-llm-r1"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.analysis_cache = {}  # 分析结果缓存

    def load_model(self):
        """加载DeepSeek-R1模型"""
        if not os.path.exists(self.model_path):
            logger.error(f"DeepSeek-R1模型路径不存在: {self.model_path}")
            return False

        try:
            logger.info("正在加载DeepSeek-R1模型...")

            # 动态导入transformers以避免不必要的依赖
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype="auto",
                load_in_4bit=True  # 使用4位量化减少内存需求
            )
            self.loaded = True
            logger.info("DeepSeek-R1模型加载成功")
            return True
        except ImportError:
            logger.error("transformers库未安装，无法加载模型")
            return False
        except Exception as e:
            logger.error(f"加载DeepSeek-R1模型失败: {str(e)}")
            return False

    def safe_generate(self, prompt, max_length=128, temperature=0.7):
        """安全的文本生成方法，带错误处理"""
        if not self.loaded:
            return "模型未加载"

        try:
            # 生成提示的哈希值作为缓存键
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

            # 检查缓存
            if prompt_hash in self.analysis_cache:
                return self.analysis_cache[prompt_hash]

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 保存到缓存
            self.analysis_cache[prompt_hash] = result
            return result
        except Exception as e:
            logger.error(f"文本生成失败: {str(e)}")
            return "分析生成失败，请检查模型配置"


class AlertAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knowledge_base = None
        self.snort_engine = None
        self.custom_rule_engine = None
        self.ml_detector = None
        self.llm_service = None
        self.initialize_components()

    def initialize_components(self):
        logger.info("正在初始化智能研判智能体...")
        self.knowledge_base = DocumentKnowledgeRetriever(
            knowledge_dir=self.config.get('knowledge_dir', 'knowledge_base')
        )
        logger.info("初始化Snort规则引擎...")
        self.snort_engine = RuleEngine()
        load_rules(self.snort_engine)
        logger.info("初始化自定义规则引擎...")
        self.custom_rule_engine = EnhancedRuleEngine(
            rule_dir=self.config.get('rule_dir', 'rules')
        )
        logger.info("初始化机器学习检测引擎...")
        self.ml_detector = AnomalyDetector()
        self.ml_detector.initialize_ml_engines()

        # 按需初始化DeepSeek-R1
        if self.config.get('use_llm', True):
            logger.info("初始化DeepSeek-R1 LLM服务...")
            self.llm_service = DeepSeekR1Service(
                model_path=self.config.get('llm_model_path', 'llm_models/deepseek-llm-r1')
            )
            # 尝试加载模型，如果失败则禁用LLM
            if not self.llm_service.load_model():
                logger.warning("DeepSeek-R1加载失败，将禁用LLM分析功能")
                self.llm_service = None

        logger.info("智能研判智能体初始化完成")

    def analyze_pcap(self, pcap_file: str) -> Dict[str, Any]:
        if not os.path.exists(pcap_file):
            logger.error(f"PCAP文件不存在: {pcap_file}")
            return {"error": "PCAP file not found"}

        start_time = time.time()
        logger.info(f"开始分析PCAP文件: {pcap_file}")

        # 1. Snort规则分析
        logger.info("运行Snort规则引擎分析...")
        snort_results = snort_analyze_pcap(pcap_file, self.snort_engine)

        # 2. 自定义规则分析
        logger.info("运行自定义规则引擎分析...")
        custom_alerts = RuleEnginePcapAnalyzer.analyze_pcap(pcap_file)
        custom_results = []
        for alert in custom_alerts:
            matches = self.custom_rule_engine.match(alert)
            if matches:
                custom_results.append({
                    "alert": alert,
                    "matches": matches
                })

        # 3. 机器学习分析
        logger.info("运行机器学习引擎分析...")
        ml_results = self.ml_detector.analyze_pcap(pcap_file)

        # 4. 整合分析结果
        logger.info("整合分析结果...")
        consolidated_results = self.consolidate_results(
            snort_results,
            custom_results,
            ml_results
        )

        # 5. 知识增强分析
        logger.info("进行知识增强分析...")
        enriched_results = self.enrich_with_knowledge(consolidated_results)

        # 6. LLM智能分析（仅处理高风险告警）
        if self.llm_service and self.llm_service.loaded:
            logger.info("进行DeepSeek-R1智能分析（仅高风险告警）...")
            high_risk_results = [res for res in enriched_results if res.get('severity', 3) <= 2]
            logger.info(f"发现 {len(high_risk_results)} 个高风险告警进行深度分析")

            # 逐个生成分析结果，确保可靠性
            for result in high_risk_results:
                prompt = self.generate_analysis_prompt(result)
                result["llm_analysis"] = self.llm_service.safe_generate(prompt)

                # 添加生成标记，避免重复生成
                result["llm_generated"] = True

        # 7. 生成最终报告
        report = self.generate_report(
            pcap_file,
            enriched_results,
            time.time() - start_time
        )

        logger.info(f"分析完成，耗时: {time.time() - start_time:.2f}秒")
        return report

    def generate_analysis_prompt(self, result: Dict) -> str:
        """生成优化的分析提示词，包含具体攻击类型"""
        # 提取具体的攻击类型
        attack_type = result.get('classification', '未知攻击')
        if 'details' in result and 'prediction' in result['details']:
            attack_type = result['details']['prediction']

        return f"""
        ## 告警深度分析请求
        **攻击类型**: {attack_type}
        **来源系统**: {result.get('source', '未知')}
        **严重程度**: {result.get('severity', '未知')}
        **事件描述**: {result.get('msg', '无描述信息')}
        **源IP**: {result.get('src_ip', '未知')}
        **目标IP**: {result.get('dst_ip', '未知')}
        **协议**: {result.get('protocol', '未知')}
    
        请针对此攻击类型提供专业分析:
        1. 攻击原理与技术特点（50字以内）
        2. 典型攻击场景与目标（30字以内）
        3. 关键防御建议（40字以内）
        """

    def consolidate_results(
            self,
            snort_results: List[Dict],
            custom_results: List[Dict],
            ml_results: Dict
    ) -> List[Dict]:
        """
        整合来自不同引擎的分析结果

        参数:
            snort_results: Snort引擎结果
            custom_results: 自定义规则引擎结果
            ml_results: 机器学习引擎结果

        返回:
            整合后的结果列表
        """
        consolidated = []

        # 处理Snort结果
        for alert in snort_results:
            consolidated.append({
                "type": "snort",
                "source": "Snort规则引擎",
                "severity": alert.get("severity", 3),
                "classification": alert.get("classification", "Unknown"),
                "msg": alert.get("msg", "No message"),
                "src_ip": alert.get("src_ip", ""),
                "dst_ip": alert.get("dst_ip", ""),
                "protocol": alert.get("protocol", ""),
                "timestamp": alert.get("timestamp", ""),
                "details": alert
            })

        # 处理自定义规则结果
        for result in custom_results:
            alert = result["alert"]
            matches = result["matches"]

            # 取最高严重级别的匹配
            max_severity = 3
            for match in matches:
                severity_map = {
                    "critical": 1,
                    "high": 2,
                    "medium": 3,
                    "low": 4,
                    "info": 5
                }
                severity = severity_map.get(match.get("severity", "medium").lower(), 3)
                if severity < max_severity:
                    max_severity = severity

            consolidated.append({
                "type": "custom_rule",
                "source": "自定义规则引擎",
                "severity": max_severity,
                "classification": matches[0]["rule_name"] if matches else "Custom Rule Alert",
                "msg": matches[0]["description"] if matches else "Custom rule triggered",
                "src_ip": alert.get("src_ip", ""),
                "dst_ip": alert.get("dst_ip", ""),
                "protocol": alert.get("proto", ""),
                "timestamp": alert.get("start_time", ""),
                "details": result
            })

        # 只要出现「正常流量」且置信度>=0.9 就一律标为正常（severity=5），绝不标严重/高危/中危
        def _is_normal_traffic(a):
            pred = str(a.get("prediction") or "")
            if pred == "正常流量" or pred.strip() == "正常流量":
                return True
            details = a.get("attack_details") or []
            for d in details:
                t = str(d.get("type") or "")
                c = float(d.get("confidence") or 0)
                if ("正常" in t or t == "正常流量") and c >= 0.9:
                    return True
            if not details and "未知攻击类型" in pred:
                return True
            # 兜底：按 attack_details 拼出「可能类型」描述，若含 正常流量(1.00) 或 正常流量(1.0) 则视为正常
            try:
                parts = [f"{d.get('type', '')}({float(d.get('confidence', 0)):.2f})" for d in details]
                combined = " ".join(parts)
                if "正常流量(1.00)" in combined or "正常流量(1.0)" in combined:
                    return True
            except Exception:
                pass
            return False

        for anomaly in ml_results.get("ml_based_predictions", []):
            if _is_normal_traffic(anomaly):
                consolidated.append({
                    "type": "ml",
                    "source": f"ML引擎 ({anomaly['model_type']} - {anomaly['dataset']})",
                    "severity": 5,
                    "classification": "正常流量",
                    "msg": "正常流量",
                    "src_ip": anomaly.get("src_ip", ""),
                    "dst_ip": anomaly.get("dst_ip", ""),
                    "protocol": anomaly.get("protocol", ""),
                    "timestamp": "",
                    "details": anomaly
                })
                continue
            # 仅当存在“有意义的”攻击类型（非“未知”且置信度>0）时才视为具体类型
            attack_details = anomaly.get("attack_details") or []
            meaningful_details = [
                d for d in attack_details
                if d.get("confidence", 0) > 0
                and "未知" not in str(d.get("type", ""))
            ]
            if meaningful_details:
                attack_description = anomaly["prediction"]
                details = [f"{d['type']}({d['confidence']:.2f})" for d in meaningful_details]
                attack_description += f" [可能类型: {', '.join(details)}]"
            else:
                attack_description = "异常流量（未识别具体类型）"

            # 确定严重程度：仅当有具体类型或规则命中时提高；纯“异常”降为中危
            if "DDoS" in attack_description or "拒绝服务" in attack_description:
                severity = 1
            elif "暴力破解" in attack_description or "SQL注入" in attack_description:
                severity = 1
            elif "扫描" in attack_description:
                severity = 2
            elif meaningful_details:
                severity = 2 if anomaly["confidence"] > 0.9 else 3
            else:
                severity = 3

            # 二次检查：描述里若「正常流量(1.00)」或「正常流量(1.0)」出现，一律改为正常，不标危
            if "正常流量(1.00)" in attack_description or "正常流量(1.0)" in attack_description:
                severity = 5
                attack_description = "正常流量"
                msg = "正常流量"

            if meaningful_details and severity != 5:
                msg = f"检测到 {anomaly['prediction']} (置信度: {anomaly['confidence']:.2f})"
            elif severity != 5:
                msg = f"检测到异常流量 (置信度: {anomaly['confidence']:.2f})，模型未给出具体攻击类型"

            consolidated.append({
                "type": "ml",
                "source": f"ML引擎 ({anomaly['model_type']} - {anomaly['dataset']})",
                "severity": severity,
                "classification": attack_description,
                "msg": msg,
                "src_ip": anomaly.get("src_ip", ""),
                "dst_ip": anomaly.get("dst_ip", ""),
                "protocol": anomaly.get("protocol", ""),
                "timestamp": "",
                "details": anomaly
            })

        # 按严重程度排序
        consolidated.sort(key=lambda x: x["severity"])
        return consolidated

    def enrich_with_knowledge(self, results: List[Dict]) -> List[Dict]:
        """
        使用知识库增强分析结果

        参数:
            results: 整合后的结果列表

        返回:
            知识增强后的结果列表
        """
        enriched_results = []

        for result in results:
            # 根据分类信息检索相关知识
            knowledge = {}

            # 尝试从消息中提取技术ID（如MITRE ATT&CK技术）
            technique_ids = re.findall(r'T\d{4}', result.get("msg", ""))
            if technique_ids:
                for tid in technique_ids:
                    attack_chain = self.knowledge_base.get_attack_chain(tid)
                    knowledge[tid] = {
                        "attack_chain": attack_chain
                    }

            # 根据威胁类型检索响应建议
            threat_types = []
            if "malware" in result["classification"].lower():
                threat_types.append("malware")
            if "exploit" in result["classification"].lower():
                threat_types.append("exploit")
            if "ransomware" in result["classification"].lower():
                threat_types.append("ransomware")
            if "dos" in result["classification"].lower():
                threat_types.append("dos")

            if not threat_types:
                # 尝试从消息中提取关键词
                keywords = ["attack", "vulnerability", "exploit", "malware", "intrusion"]
                for kw in keywords:
                    if kw in result["msg"].lower():
                        threat_types.append(kw)
                        break

            response_advice = []
            for threat in threat_types:
                advice = self.knowledge_base.get_incident_response(threat)
                response_advice.append({
                    "threat_type": threat,
                    "advice": advice
                })

            # 添加知识到结果
            enriched_result = result.copy()
            enriched_result["knowledge"] = {
                "techniques": knowledge,
                "response_advice": response_advice
            }

            enriched_results.append(enriched_result)

        return enriched_results

    def enhance_with_llm(self, results: List[Dict]) -> List[Dict]:
        """
        使用DeepSeek-R1 LLM增强分析结果

        参数:
            results: 知识增强后的结果列表

        返回:
            LLM增强后的结果列表
        """
        enhanced_results = []

        for result in results:
            enhanced_result = result.copy()

            # 使用LLM生成威胁分析
            threat_analysis = self.generate_threat_analysis(result)
            enhanced_result["llm_analysis"] = threat_analysis

            # 使用LLM生成响应建议
            response_suggestions = self.generate_response_suggestions(result)
            enhanced_result["llm_response_suggestions"] = response_suggestions

            enhanced_results.append(enhanced_result)

        return enhanced_results

    def generate_threat_analysis(self, result: Dict) -> str:
        """
        使用DeepSeek-R1生成威胁分析

        参数:
            result: 单个告警结果

        返回:
            LLM生成的威胁分析文本
        """
        if not self.llm_service:
            return "LLM服务未启用"

        prompt = f"""
        你是一名网络安全分析师，请对以下安全告警进行深入分析：

        告警类型: {result.get('classification', '未知')}
        来源: {result.get('source', '未知')}
        严重程度: {result.get('severity', '未知')}
        描述: {result.get('msg', '无描述信息')}
        源IP: {result.get('src_ip', '未知')}
        目标IP: {result.get('dst_ip', '未知')}
        协议: {result.get('protocol', '未知')}
        时间: {result.get('timestamp', '未知')}

        请分析:
        1. 该告警可能对应的攻击类型（如APT攻击、DDoS、勒索软件等）
        2. 攻击者的可能意图
        3. 该攻击在ATT&CK框架中的技术映射
        4. 潜在的攻击链分析
        5. 对目标系统可能造成的影响

        请用专业但易懂的语言撰写分析报告。
        """

        return self.llm_service.generate(prompt)

    def generate_response_suggestions(self, result: Dict) -> str:
        """
        使用DeepSeek-R1生成响应建议

        参数:
            result: 单个告警结果

        返回:
            LLM生成的响应建议文本
        """
        if not self.llm_service:
            return "LLM服务未启用"

        prompt = f"""
        你是一名网络安全响应专家，请根据以下安全告警提供专业的响应建议：

        告警类型: {result.get('classification', '未知')}
        严重程度: {result.get('severity', '未知')}
        源IP: {result.get('src_ip', '未知')}
        目标IP: {result.get('dst_ip', '未知')}
        协议: {result.get('protocol', '未知')}

        请提供:
        1. 立即采取的应急措施
        2. 受影响系统的隔离和取证建议
        3. 长期防护策略建议
        4. 相关的威胁情报查询建议
        5. 需要收集的日志和证据类型

        请用清晰的结构化格式（带编号）提供建议。
        """

        return self.llm_service.generate(prompt)

    def generate_report(self, pcap_file: str, results: List[Dict], duration: float) -> Dict[str, Any]:
        """
        生成综合研判报告

        参数:
            pcap_file: 分析的PCAP文件
            results: 知识增强后的结果
            duration: 分析耗时

        返回:
            综合研判报告字典
        """
        # 兜底：凡 classification 为「正常流量」的一律标为正常（severity=5），避免被统计/展示为中危
        for r in results:
            if (r.get("classification") or "").strip() == "正常流量":
                r["severity"] = 5

        # 统计摘要（severity 1=严重 2=高危 3=中危 5=正常）
        total_alerts = len(results)
        critical_alerts = sum(1 for r in results if r["severity"] == 1)
        high_alerts = sum(1 for r in results if r["severity"] == 2)
        medium_alerts = sum(1 for r in results if r["severity"] == 3)
        normal_alerts = sum(1 for r in results if r["severity"] == 5)

        # 威胁分类统计
        threat_categories = {}
        attack_type_count = {}

        for result in results:
            # 主分类
            category = result["classification"].split(":")[0].strip()
            threat_categories[category] = threat_categories.get(category, 0) + 1

            # 详细攻击类型统计
            if "ml" in result.get("type", "") and "details" in result:
                attack_type = result["details"].get("prediction", "未知攻击")
                attack_type_count[attack_type] = attack_type_count.get(attack_type, 0) + 1

        report = {
            "metadata": {
                "pcap_file": os.path.basename(pcap_file),
                "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(duration, 2),
                "analysis_system": "DeepSeek-R1 Alert Agent"
            },
            "summary": {
                "total_alerts": len(results),
                "critical_alerts": sum(1 for r in results if r["severity"] == 1),
                "high_alerts": sum(1 for r in results if r["severity"] == 2),
                "medium_alerts": sum(1 for r in results if r["severity"] == 3),
                "normal_alerts": sum(1 for r in results if r["severity"] == 5),
                "threat_categories": threat_categories,
                "attack_types": attack_type_count
            },
            "detailed_results": results,
            "recommendations": self.generate_recommendations(results)
        }

        return report

    def generate_recommendations(self, results: List[Dict]) -> List[str]:
        """生成安全建议（优化版）"""
        recommendations = []
        critical_threats = set()

        # 收集所有关键威胁
        for result in results:
            if result["severity"] <= 2:  # Critical或High
                critical_threats.add(result["classification"].split(":")[0].strip())

                # 如果有LLM分析结果，优先使用
                if result.get("llm_response_suggestions"):
                    recommendations.append(result["llm_response_suggestions"])
                elif "knowledge" in result and "response_advice" in result["knowledge"]:
                    for advice in result["knowledge"]["response_advice"]:
                        for item in advice["advice"]:
                            if item not in recommendations:
                                recommendations.append(item)

        # 如果没有关键威胁，添加一般性建议
        if not recommendations:
            recommendations.append("未检测到高风险威胁，建议进行常规安全审计")
        else:
            # 添加基于威胁类型的建议
            if "Malware" in critical_threats:
                recommendations.append("检测到恶意软件活动，建议进行全系统扫描和隔离感染主机")
            if "Exploit" in critical_threats:
                recommendations.append("检测到漏洞利用尝试，建议立即修补相关系统和应用")
            if "DDoS" in critical_threats:
                recommendations.append("检测到DDoS攻击迹象，建议启用流量清洗和防火墙规则")

        # 添加通用建议
        recommendations.append("建议审查所有警报的原始数据包以确认攻击行为")
        recommendations.append("建议更新所有安全规则和签名数据库")
        recommendations.append("建议对受影响系统进行深度取证分析")

        return recommendations[:10]  # 最多返回10条建议

    def save_report(self, report: Dict, output_dir: str = "reports"):
        """保存分析报告到文件"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"alert_report_{report['metadata']['pcap_file']}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"分析报告已保存至: {filepath}")
        return filepath


def main():
    """主函数，演示智能体的使用"""
    # 配置智能体
    config = {
        "knowledge_dir": "knowledge_base",
        "rule_dir": "rules",
        "use_llm": True,  # 启用DeepSeek-R1
        "llm_model_path": "llm_models/deepseek-llm-r1"  # 模型路径
    }

    # 创建智能体实例
    agent = AlertAgent(config)

    # 分析PCAP文件
    pcap_file = "sample.pcap"  # 替换为实际PCAP文件路径
    report = agent.analyze_pcap(pcap_file)

    # 保存报告
    report_path = agent.save_report(report)

    print(f"\n{'=' * 50}")
    print("告警智能研判报告摘要:")
    print(f"PCAP文件: {report['metadata']['pcap_file']}")
    print(f"分析时间: {report['metadata']['analysis_time']}")
    print(f"分析耗时: {report['metadata']['duration_seconds']}秒")
    print(f"总告警数: {report['summary']['total_alerts']}")
    print(f"严重告警: {report['summary']['critical_alerts']}")
    print(f"高危告警: {report['summary']['high_alerts']}")

    print("\n威胁分类统计:")
    for category, count in report['summary']['threat_categories'].items():
        print(f"  - {category}: {count}")

    print("\n攻击类型分布:")
    for attack_type, count in report['summary'].get('attack_types', {}).items():
        print(f"  - {attack_type}: {count}")

    # 显示LLM分析摘要
    if any("llm_analysis" in res for res in report.get('detailed_results', [])):
        print("\nDeepSeek-R1分析摘要:")
        llm_results = [res for res in report['detailed_results'] if "llm_analysis" in res]
        for i, res in enumerate(llm_results[:3], 1):
            analysis = res["llm_analysis"]
            print(f"告警 #{i} ({res['classification']}):")
            # 确保分析结果不为空
            if analysis.strip():
                print(analysis)
            else:
                print("未生成有效分析")
            print("-" * 50)

    print("\n关键建议:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"  {i}. {rec}")

    print(f"\n完整报告已保存至: {report_path}")
    print('=' * 50)

if __name__ == "__main__":
    main()