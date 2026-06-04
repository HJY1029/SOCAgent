import json
import re
import os
import logging
import time
import traceback
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM

from engine.ids_analyzer import RuleEngine, load_rules, analyze_pcap
from engine.ml_engine import MLEngine
from engine.rule_engine import EnhancedRuleEngine, PcapAnalyzer  # 添加自定义规则引擎
from knowledge_retriever import DocumentKnowledgeRetriever

# 配置更详细的日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("security_agent.log")
    ]
)
logger = logging.getLogger("SecurityAgent")


class SecurityAgent:
    def __init__(self, model_name="llm_models/deepseek-llm-r1"):
        # 初始化组件
        self.rule_engine = RuleEngine()
        self.custom_rule_engine = EnhancedRuleEngine()
        self.ml_engines = self._init_ml_engines()
        self.knowledge = DocumentKnowledgeRetriever()

        # 初始化大模型
        logging.info("加载DeepSeek模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            pad_token="<|endoftext|>"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            pad_token_id=self.tokenizer.pad_token_id
        )
        logging.info("模型加载完成")

        # 工具定义（优化描述）
        self.tools = [
            {
                "name": "rule_engine",
                "description": "使用Snort规则库检测已知攻击模式",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pcap_path": {"type": "string", "description": "PCAP文件路径"}
                    },
                    "required": ["pcap_path"]
                }
            },
            {
                "name": "custom_rule_engine",
                "description": "使用自定义规则检测引擎分析高级威胁",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pcap_path": {"type": "string", "description": "PCAP文件路径"}
                    },
                    "required": ["pcap_path"]
                }
            },
            {
                "name": "ml_model",
                "description": "使用AI模型检测未知威胁和异常行为",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pcap_path": {"type": "string", "description": "PCAP文件路径"},
                        "model_type": {
                            "type": "string",
                            "description": "模型类型(isolation_forest/xgboost/lightgbm/random_forest)",
                            "default": "xgboost"
                        },
                        "dataset_type": {
                            "type": "string",
                            "description": "数据集类型(kdd99/cicids2017)",
                            "default": "cicids2017"
                        }
                    },
                    "required": ["pcap_path"]
                }
            },
            {
                "name": "knowledge_retriever",
                "description": "获取威胁情报、攻击链分析和响应建议",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "doc_type": {
                            "type": "string",
                            "description": "文档类型(attack_framework/threat_intel/threat_case/response_playbook)",
                            "default": "attack_framework"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

        # 工具依赖关系
        self.tool_dependencies = {
            "knowledge_retriever": ["rule_engine", "custom_rule_engine", "ml_model"]
        }

        # 系统提示词（强化工具调用要求）
        self.system_prompt = self._create_system_prompt()
        self.max_retries = 3  # 工具调用重试次数
        self.required_tools = ["rule_engine", "custom_rule_engine", "ml_model"]  # 必须调用的核心工具

    def _init_ml_engines(self):
        """初始化所有机器学习引擎（带错误处理）"""
        ml_engines = {}
        model_types = ["isolation_forest", "xgboost", "lightgbm", "random_forest"]
        dataset_types = ["kdd99", "cicids2017"]

        for model_type in model_types:
            for dataset_type in dataset_types:
                key = f"{model_type}_{dataset_type}"
                try:
                    # 尝试初始化引擎
                    ml_engines[key] = MLEngine(model_type, dataset_type)

                    # 检查是否初始化成功
                    if not ml_engines[key].is_initialized():
                        logging.warning(f"ML引擎 {key} 初始化失败")
                        del ml_engines[key]
                except Exception as e:
                    # 更友好的错误处理
                    error_msg = f"初始化ML引擎 {key} 失败: {str(e)}"
                    logging.error(error_msg)

                    # 创建虚拟引擎提供错误信息
                    ml_engines[key] = {
                        "error": error_msg,
                        "is_initialized": lambda: False
                    }

        return ml_engines

    def _create_system_prompt(self):
        return """
        # 网络安全分析智能体 - 指令

        ## 输出格式要求
        你只能输出以下两种JSON格式之一：

        A. 工具调用:
        {
            "tool": "工具名",
            "parameters": {
                "参数名": "值"
            }
        }

        B. 最终报告:
        {
            "analysis_summary": "摘要",
            "threat_level": "高/中/低",
            "detected_threats": [
                {
                    "type": "威胁类型",
                    "source": "检测来源",
                    "confidence": 数值,
                    "description": "描述"
                }
            ],
            "attack_chain": ["攻击步骤1", "攻击步骤2"],
            "recommendations": {
                "immediate": ["建议1", "建议2"],
                "long_term": ["长期建议1", "长期建议2"]
            }
        }

        ## 可用工具
        1. rule_engine: 使用Snort规则检测已知攻击
           - 参数: pcap_path (必须)

        2. custom_rule_engine: 自定义规则检测高级威胁
           - 参数: pcap_path (必须)

        3. ml_model: 机器学习检测未知威胁
           - 参数: 
               - pcap_path (必须)
               - model_type (可选, 默认: xgboost)
               - dataset_type (可选, 默认: cicids2017)

        4. knowledge_retriever: 获取威胁情报
           - 参数:
               - query (必须)
               - doc_type (可选, 默认: attack_framework)

        ## 分析流程
        1. 首先调用 rule_engine
        2. 然后调用 custom_rule_engine
        3. 最后调用 ml_model

        ## 当前任务
        分析PCAP文件: sample.pcap
        """

    # 以下方法保持不变（_parse_agent_response, _check_dependencies, _generate_model_response）
    # ... [保持原有_parse_agent_response, _check_dependencies, _generate_model_response方法] ...

    def _execute_tool(self, tool_name, parameters):
        """执行工具调用并记录详细日志（带参数验证）"""
        # 验证必需参数
        required_params = {
            "rule_engine": ["pcap_path"],
            "custom_rule_engine": ["pcap_path"],
            "ml_model": ["pcap_path"],
            "knowledge_retriever": ["query"]
        }

        if tool_name in required_params:
            for param in required_params[tool_name]:
                if param not in parameters:
                    error_msg = f"缺少必需参数: {param}"
                    logger.error(error_msg)
                    return {"error": error_msg}

        # 记录调用开始
        start_time = time.time()
        logger.info(f"🛠️ 开始执行工具: {tool_name}")
        logger.info(f"🔧 工具参数: {json.dumps(parameters, indent=2)}")

        # 确保PCAP路径传递给所有需要它的工具
        if tool_name in ["rule_engine", "custom_rule_engine", "ml_model"]:
            pcap_path = parameters.get("pcap_path", "")
            if not os.path.exists(pcap_path):
                error_msg = f"PCAP文件不存在: {pcap_path}"
                logger.error(error_msg)
                return {"error": error_msg}

        try:
            result = None
            # Snort规则引擎工具
            if tool_name == "rule_engine":
                pcap_path = parameters["pcap_path"]
                logger.info(f"📦 加载Snort规则并分析PCAP: {pcap_path}")
                load_rules(self.rule_engine)
                alerts = analyze_pcap(pcap_path, self.rule_engine)
                logger.info(f"📊 Snort规则引擎检测到 {len(alerts)} 条告警")

                # 简化输出
                result = [{
                    "timestamp": alert["timestamp"],
                    "src_ip": alert["src_ip"],
                    "dst_ip": alert["dst_ip"],
                    "msg": alert["msg"],
                    "severity": alert["severity"]
                } for alert in alerts]

            # 自定义规则引擎工具
            elif tool_name == "custom_rule_engine":
                pcap_path = parameters["pcap_path"]
                logger.info(f"🛡️ 使用自定义规则分析PCAP: {pcap_path}")
                alerts = self.custom_rule_engine.match_pcap(pcap_path)
                logger.info(f"🔍 自定义规则引擎检测到 {len(alerts)} 条告警")

                # 简化输出
                result = [{
                    "id": alert["id"],
                    "src_ip": alert["src_ip"],
                    "dst_ip": alert["dst_ip"],
                    "proto": alert["proto"],
                    "severity": alert.get("severity", "medium"),
                    "matches": [match["rule_name"] for match in alert.get("matches", [])]
                } for alert in alerts]

            # 机器学习模型工具
            elif tool_name == "ml_model":
                pcap_path = parameters["pcap_path"]
                model_type = parameters.get("model_type", "xgboost")
                dataset_type = parameters.get("dataset_type", "cicids2017")
                engine_key = f"{model_type}_{dataset_type}"

                # 检查引擎是否可用
                if engine_key not in self.ml_engines:
                    error_msg = f"未找到ML引擎: {engine_key}"
                    logger.error(error_msg)
                    return {"error": error_msg}

                # 检查引擎初始化状态
                if (isinstance(self.ml_engines[engine_key], dict)) and "error" in self.ml_engines[engine_key]:
                    error_msg = self.ml_engines[engine_key]["error"]
                    logger.error(f"ML引擎 {engine_key} 不可用: {error_msg}")

                    # 提供备选方案
                    available_engines = [k for k in self.ml_engines.keys()
                                         if
                                         not (isinstance(self.ml_engines[k], dict) and "error" in self.ml_engines[k])]

                    suggestion = (
                        f"ML引擎 {engine_key} 不可用。\n"
                        f"可用引擎: {', '.join(available_engines) if available_engines else '无'}\n"
                        "请尝试其他模型类型或数据集类型。"
                    )

                    return {"error": error_msg, "suggestion": suggestion}

                logger.info(f"🤖 使用 {engine_key} 分析PCAP文件: {pcap_path}")
                result = self.ml_engines[engine_key].predict(pcap_path)

                # 统计异常结果
                anomalies = [r for r in result if r.get("prediction", "") != "normal"]
                logger.info(f"📈 机器学习检测到 {len(anomalies)} 条异常流量")

            # 知识检索工具
            elif tool_name == "knowledge_retriever":
                query = parameters.get("query", "")
                if not query:
                    error_msg = "缺少查询参数"
                    logger.error(error_msg)
                    return {"error": error_msg}

                doc_type = parameters.get("doc_type", "attack_framework")
                logger.info(f"🔍 知识检索: '{query}' (类型: {doc_type})")

                # 执行知识检索
                result = self.knowledge.retrieve(query, top_k=3, doc_type=doc_type)
                logger.info(f"📚 找到 {len(result)} 条相关知识")

            # 记录执行结果
            exec_time = time.time() - start_time
            logger.info(f"✅ 工具 {tool_name} 执行成功, 耗时: {exec_time:.2f}秒")

            return result

        except Exception as e:
            # 记录完整错误信息
            exec_time = time.time() - start_time
            error_msg = f"❌ 工具 {tool_name} 执行失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg}

    def _generate_model_response(self, messages):
        """使用大模型生成响应（修复张量形状问题）"""
        try:
            # 使用更可靠的方式编码输入
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # 确保输入是二维张量 [batch_size, sequence_length]
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)  # 添加批次维度

            # 创建注意力掩码
            attention_mask = (inputs != self.tokenizer.pad_token_id).int()

            # 移动到GPU（如果可用）
            input_ids = inputs.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            # 在生成请求前添加引导指令
            guided_messages = messages.copy()
            guided_messages.append({
                "role": "user",
                "content": "请严格按要求的JSON格式输出，不要包含任何其他文本"
            })

            # 使用更可靠的生成参数
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=300,  # 减少长度限制
                do_sample=False,  # 关闭随机采样
                num_beams=2,  # 使用束搜索
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # 避免重复
            )

            # 解码响应（跳过输入部分）
            response = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # 记录原始响应
            logger.debug(f"模型原始输出: {response}")

            # 强化JSON提取逻辑
            cleaned_response = response.strip()

            # 1. 尝试直接解析
            try:
                json_obj = json.loads(cleaned_response)
                return json.dumps(json_obj)  # 重新序列化确保格式一致
            except:
                pass

            # 2. 移除JSON代码块标记
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3].strip()

            # 3. 提取可能的JSON部分
            json_match = re.search(r'\{[\s\S]*?\}', cleaned_response)
            if json_match:
                return json_match.group(0)

            # 4. 返回格式错误
            return json.dumps({"error": "模型输出格式无效"})

        except Exception as e:
            logger.error(f"模型生成失败: {str(e)}")
            return json.dumps({"error": "模型响应生成失败"})

    def _validate_model_response(self, response):
        """验证模型响应是否为有效JSON"""
        try:
            # 尝试解析为JSON
            data = json.loads(response)

            # 检查是否是工具调用
            if "tool" in data:
                return "tool" in data and "parameters" in data

            # 检查是否是最终报告
            if "analysis_summary" in data:
                required_keys = ["threat_level", "detected_threats"]
                return all(key in data for key in required_keys)

            return False
        except:
            return False


    def _parse_agent_response(self, response):
        """简化响应解析器"""
        logger.debug(f"原始响应内容: {response}")

        # 1. 尝试直接解析
        try:
            return json.loads(response)
        except:
            pass

        # 2. 提取最长的{}包裹的内容
        json_matches = re.findall(r'\{[^{}]*\}', response)
        if json_matches:
            # 选择最长的匹配项
            longest_match = max(json_matches, key=len)
            try:
                return json.loads(longest_match)
            except:
                pass

        # 3. 返回错误
        return {"error": "无法解析模型响应"}

    def _validate_response_format(self, response):
        """验证响应格式是否符合要求"""
        if not isinstance(response, dict):
            return False

        # 检查工具调用格式
        if "tool" in response:
            return "parameters" in response and isinstance(response["parameters"], dict)

        # 检查最终报告格式
        if "analysis_summary" in response:
            required_keys = {"threat_level", "detected_threats"}
            return all(key in response for key in required_keys)

        return False


    def _check_dependencies(self, tool_name, called_tools):
        """检查工具依赖是否满足"""
        dependencies = self.tool_dependencies.get(tool_name, [])
        missing = [dep for dep in dependencies if dep not in called_tools]

        if missing:
            dep_msg = (
                f"工具 '{tool_name}' 需要先调用以下工具: {', '.join(missing)}\n"
                "请先调用这些工具后再尝试。"
            )
            return False, dep_msg
        return True, ""


    def analyze_pcap(self, pcap_path):
        """分析PCAP文件（主入口）"""
        # 验证PCAP文件存在
        if not os.path.exists(pcap_path):
            return {"error": f"PCAP文件不存在: {pcap_path}"}

        # 构建初始提示
        user_prompt = f"""
        ## 分析任务
        请分析以下PCAP文件中的网络流量，检测潜在的安全威胁：
        PCAP文件路径: {pcap_path}

        请按照工作流程使用工具获取数据，然后生成最终报告。
        注意：必须调用所有三个核心检测工具（rule_engine, custom_rule_engine, ml_model）！
        仅在检测到高威胁时调用knowledge_retriever工具。
        """

        # 对话历史
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        print(f'原始消息：{messages}')

        # 最大处理步骤
        max_steps = 6
        current_step = 0
        final_result = None
        tool_errors = 0

        # 工具调用记录
        tool_calls = {tool['name']: False for tool in self.tools}
        tool_results = {}  # 存储各工具结果
        tool_history = []  # 记录详细调用历史

        # 在循环中添加错误计数器重置逻辑
        consecutive_errors = 0
        while current_step < max_steps and final_result is None and tool_errors < 3:
            current_step += 1
            logging.info(f"分析步骤 {current_step}/{max_steps}")

            # 当连续错误超过2次时重置对话历史
            if consecutive_errors >= 2:
                logger.warning("连续错误过多，重置对话历史")
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"PCAP文件路径: {pcap_path}\n请重新开始分析"}
                ]
                consecutive_errors = 0

            # 调用大模型
            response = self._generate_model_response(messages)
            logger.info(f"模型原始响应:\n{response}")

            # 解析响应
            try:
                action = json.loads(response)  # 直接尝试解析为JSON
                valid_format = True
            except json.JSONDecodeError:
                action = self._parse_agent_response(response)
                valid_format = self._validate_response_format(action)

            if not valid_format:
                consecutive_errors += 1
                error_msg = "模型响应格式无效! 请确保输出纯JSON格式"
                logger.error(error_msg)

                # 简化的错误反馈
                messages.append({
                    "role": "user",
                    "content": "ERROR: 输出格式无效！必须输出纯JSON对象，无任何额外文本"
                })
                continue

            # 重置连续错误计数器
            consecutive_errors = 0

            # 添加到消息历史
            messages.append({"role": "assistant", "content": response})

            # 解析响应
            action = self._parse_agent_response(response)
            logging.info(f"解析后的动作: {json.dumps(action, indent=2)}")

            # 处理工具调用
            if "tool" in action:
                tool_name = action["tool"]
                tool_params = action.get("parameters", {})

                # 添加依赖检查
                if tool_name in self.tool_dependencies:
                    deps_ok, dep_msg = self._check_dependencies(
                        tool_name,
                        [call["tool"] for call in tool_history]
                    )
                    if not deps_ok:
                        messages.append({"role": "user", "content": dep_msg})
                        logger.warning(dep_msg)
                        continue

                # 执行工具
                tool_result = self._execute_tool(tool_name, tool_params)

                # 存储结果
                tool_results[tool_name] = tool_result
                tool_calls[tool_name] = True
                tool_history.append({
                    "step": current_step,
                    "tool": tool_name,
                    "parameters": tool_params,
                    "result_summary": f"返回{len(tool_result)}条记录" if isinstance(tool_result, list) else "结果"
                })

                # 将结果添加到提示
                result_str = f"工具 {tool_name} 返回结果:\n{json.dumps(tool_result, indent=2)}"
                messages.append({"role": "user", "content": result_str})

                # 检查错误
                if "error" in tool_result:
                    tool_errors += 1
                    logging.warning(f"工具调用失败: {tool_result.get('error', '未知错误')}")

                    # 如果有建议，添加到消息中
                    if "suggestion" in tool_result:
                        messages.append({"role": "user", "content": tool_result["suggestion"]})

            elif "analysis_summary" in action:
                # 验证核心工具是否全部调用
                missing_tools = [tool for tool in self.required_tools if not tool_calls.get(tool, False)]
                if missing_tools:
                    error_msg = (
                        f"核心工具调用不完整！未使用的工具: {', '.join(missing_tools)}\n"
                        "请确保调用所有三个核心检测工具。"
                    )
                    messages.append({"role": "user", "content": error_msg})
                    continue

                # 最终分析结果
                final_result = action

                # 添加详细工具结果
                final_result["tool_results"] = {
                    tool: {
                        "called": tool_calls.get(tool, False),
                        "result_count": len(tool_results.get(tool, []))
                    } for tool in self.tools
                }

                # 添加工具使用历史
                final_result["tool_history"] = tool_history

                # 融合多引擎结果
                final_result = self._fuse_results(final_result, tool_results)

            else:
                # 错误处理
                error_msg = f"无法解析响应: {action.get('error', '未知错误')}"
                messages.append({"role": "user", "content": error_msg})

        if final_result is None:
            return {
                "error": "分析失败",
                "reason": "超过最大处理步骤或工具调用失败过多",
                "steps": current_step,
                "tool_errors": tool_errors,
                "tool_history": tool_history,
                "tool_status": tool_calls
            }

        return final_result

    def _fuse_results(self, final_result, tool_results):
        """融合多个检测引擎的结果"""
        # 从各工具提取关键发现
        threats = []

        # 从Snort规则引擎提取
        snort_results = tool_results.get("rule_engine", [])
        for alert in snort_results:
            threats.append({
                "type": "已知攻击模式",
                "source": "Snort规则引擎",
                "confidence": 90,  # Snort规则置信度较高
                "description": alert.get("msg", ""),
                "severity": alert.get("severity", "medium")
            })

        # 从自定义规则引擎提取
        custom_results = tool_results.get("custom_rule_engine", [])
        for alert in custom_results:
            for match in alert.get("matches", []):
                threats.append({
                    "type": "高级威胁",
                    "source": "自定义规则引擎",
                    "confidence": 85,
                    "description": match,
                    "severity": alert.get("severity", "medium")
                })

        # 从机器学习引擎提取
        ml_results = tool_results.get("ml_model", [])
        for anomaly in ml_results:
            if anomaly.get("prediction", "") != "normal":
                threats.append({
                    "type": "异常行为",
                    "source": "机器学习引擎",
                    "confidence": int(anomaly.get("confidence", 0.7) * 100),
                    "description": f"{anomaly.get('prediction', '异常')}行为检测",
                    "severity": "high" if anomaly.get("confidence", 0) > 0.8 else "medium"
                })

        # 更新最终结果
        final_result["detected_threats"] = threats

        # 确定整体威胁级别
        threat_levels = [t.get("severity", "low") for t in threats]
        if "high" in threat_levels:
            final_result["threat_level"] = "高"
        elif "medium" in threat_levels:
            final_result["threat_level"] = "中"
        else:
            final_result["threat_level"] = "低"

        # 如果存在高威胁，添加知识库建议
        if final_result["threat_level"] == "高":
            top_threat = max(threats, key=lambda x: x.get("confidence", 0), default=None)
            if top_threat:
                attack_chain = self.knowledge.get_attack_chain(top_threat["description"])
                response_plan = self.knowledge.get_incident_response(top_threat["type"])

                final_result["attack_chain"] = attack_chain
                final_result["recommendations"] = {
                    "immediate": response_plan[:3],  # 取前3条立即行动
                    "long_term": response_plan[3:]  # 其余作为长期建议
                }

        return final_result


# 使用示例
if __name__ == "__main__":
    agent = SecurityAgent()

    # 测试分析PCAP文件
    pcap_path = "sample.pcap"

    print("开始安全分析...")
    start_time = time.time()

    result = agent.analyze_pcap(pcap_path)

    duration = time.time() - start_time
    print(f"\n分析完成，耗时: {duration:.2f}秒")
    print("最终分析结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))