import json
import operator
import os
import re
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import rdpcap
import sys
from collections import defaultdict
import time
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('pcap_analyzer')


class EnhancedRuleEngine:
    OPERATORS = {
        'eq': operator.eq,
        'ne': operator.ne,
        'gt': operator.gt,
        'ge': operator.ge,
        'lt': operator.lt,
        'le': operator.le,
        'contains': lambda a, b: b in a if isinstance(a, str) else False,
        'icontains': lambda a, b: b.lower() in a.lower() if isinstance(a, str) else False,
        'matches': lambda a, b: re.match(b, a) is not None if isinstance(a, str) else False,
        'in': lambda a, b: a in b if hasattr(b, '__contains__') else False,
        'not_in': lambda a, b: a not in b if hasattr(b, '__contains__') else False
    }

    def __init__(self, rule_dir="rules"):
        self.rules = self._load_rules(rule_dir)
        logger.info(f"规则引擎初始化完成，加载 {len(self.rules)} 条规则")

    def _load_rules(self, rule_dir):
        """从目录加载所有JSON规则文件"""
        rules = []

        if not os.path.exists(rule_dir):
            logger.error(f"规则目录 {rule_dir} 不存在")
            return rules

        for file in os.listdir(rule_dir):
            if file.endswith('.json'):
                file_path = os.path.join(rule_dir, file)
                try:
                    # 显式指定UTF-8编码解决GBK问题
                    with open(file_path, encoding='utf-8') as f:
                        file_rules = json.load(f)
                        if not isinstance(file_rules, list):
                            logger.warning(f"规则文件 {file} 格式错误: 顶层结构应为列表")
                            continue
                        rules.extend(file_rules)
                        logger.info(f"已加载规则文件: {file} ({len(file_rules)} 条规则)")
                except Exception as e:
                    logger.error(f"加载规则文件 {file} 失败: {str(e)}")
                    logger.debug("详细错误信息:", exc_info=True)

        logger.info(f"共加载 {len(rules)} 条规则")
        return rules

    def _evaluate_condition(self, condition, alert):
        """评估单个条件"""
        try:
            field_path = condition['field'].split('.')
            current = alert

            # 遍历嵌套字段
            for part in field_path:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                    current = current[int(part)]
                else:
                    return False  # 字段不存在

            op_func = self.OPERATORS.get(condition['operator'])
            if not op_func:
                logger.warning(f"未知运算符: {condition['operator']}")
                return False

            # 获取比较值
            value = condition['value']

            # 处理列表类型的值
            if isinstance(value, list) and condition['operator'] in ['contains', 'icontains', 'in', 'not_in']:
                # 对于列表操作，使用列表中的每个元素进行测试
                if condition['operator'] in ['contains', 'icontains']:
                    # contains 和 icontains 检查是否包含任意一个值
                    for v in value:
                        try:
                            if op_func(current, v):
                                return True
                        except Exception:
                            pass
                    return False
                else:
                    # in 和 not_in 使用整个列表作为参数
                    return op_func(current, value)

            # 类型转换尝试
            try:
                if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                    value = float(value)
                if isinstance(current, str) and current.replace('.', '', 1).isdigit():
                    current = float(current)
            except:
                pass

            # 执行比较
            try:
                return op_func(current, value)
            except Exception as e:
                logger.debug(f"比较失败: {current} {condition['operator']} {value}, 错误: {str(e)}")
                # 类型不匹配时尝试字符串比较
                try:
                    return op_func(str(current), str(value))
                except:
                    return False
        except Exception as e:
            logger.error(f"评估条件失败: {condition}, 错误: {str(e)}")
            return False

    def match(self, alert):
        """匹配告警规则"""
        results = []
        for rule in self.rules:
            try:
                condition_results = []
                condition_details = []  # 保存每个条件的详细信息

                for cond in rule.get('conditions', []):
                    result = self._evaluate_condition(cond, alert)
                    condition_results.append(result)
                    # 保存条件详情用于调试
                    condition_details.append({
                        "field": cond['field'],
                        "operator": cond['operator'],
                        "value": cond['value'],
                        "result": result
                    })

                # 应用逻辑运算符
                if rule.get('logic', 'all') == 'all':
                    match = all(condition_results)
                else:  # 'any'
                    match = any(condition_results)

                if match:
                    results.append({
                        "rule_id": rule['id'],
                        "rule_name": rule['name'],
                        "severity": rule.get('severity', 'medium'),
                        "description": rule['description'],
                        "action": rule.get('action', 'alert'),
                        "condition_details": condition_details  # 包含条件详情
                    })
            except Exception as e:
                logger.error(f"处理规则 {rule.get('id', 'unknown')} 失败: {str(e)}")
        return results


class PcapAnalyzer:
    FLOW_TIMEOUT = 60.0  # 60秒流超时时间

    @staticmethod
    def _get_flow_key(packet):
        """生成流的唯一标识键（五元组）"""
        if IP in packet:
            src = packet[IP].src
            dst = packet[IP].dst
            proto = packet[IP].proto

            if TCP in packet:
                return (src, dst, packet[TCP].sport, packet[TCP].dport, proto)
            elif UDP in packet:
                return (src, dst, packet[UDP].sport, packet[UDP].dport, proto)
        return None

    @staticmethod
    def _get_payload(packet):
        """提取数据包的有效载荷"""
        if TCP in packet:
            return bytes(packet[TCP].payload)
        elif UDP in packet:
            return bytes(packet[UDP].payload)
        return b""

    @staticmethod
    def analyze_pcap(pcap_file):
        """分析PCAP文件并生成告警数据流"""
        try:
            logger.info(f"开始分析PCAP文件: {pcap_file}")
            packets = rdpcap(pcap_file)
            logger.info(f"成功读取 {len(packets)} 个数据包")
        except Exception as e:
            logger.error(f"读取PCAP文件失败: {str(e)}")
            return []

        flows = defaultdict(list)
        current_flows = {}

        # 分组数据包到流（支持流超时）
        for packet in packets:
            flow_key = PcapAnalyzer._get_flow_key(packet)
            if not flow_key:
                continue

            # 处理流超时
            if flow_key in current_flows:
                last_packet = current_flows[flow_key][-1]
                if packet.time - last_packet.time > PcapAnalyzer.FLOW_TIMEOUT:
                    # 超时，将当前流归档
                    flows[flow_key].append(current_flows[flow_key])
                    current_flows[flow_key] = []

            current_flows.setdefault(flow_key, []).append(packet)

        # 添加剩余的流
        for flow_key, packets_list in current_flows.items():
            flows[flow_key].append(packets_list)

        alerts = []
        total_flows = sum(len(flow_list) for flow_list in flows.values())
        logger.info(f"发现 {total_flows} 个网络流")

        # 为每个流生成告警
        for flow_key, flow_list in flows.items():
            for i, packets in enumerate(flow_list):
                if not packets:
                    continue

                # 排序数据包确保时间顺序
                packets.sort(key=lambda p: p.time)

                # 提取流基本信息
                first_pkt = packets[0]
                last_pkt = packets[-1]
                src_ip, dst_ip, sport, dport, proto = flow_key

                # 计算流持续时间
                flow_duration = last_pkt.time - first_pkt.time

                # 初始化特征字典
                features = {
                    "Flow Duration": flow_duration,
                    "SYN Flag Count": 0,
                    "RST Flag Count": 0,
                    "Total Fwd Packets": 0,
                    "Total Bwd Packets": 0,
                    "Fwd Packet Length Max": 0,
                    "Bwd Packet Length Max": 0,
                    "Total Bytes": 0,
                    "Flow Bytes/s": 0,
                }

                # 拼接原始数据
                raw_data = b""

                # 分析每个数据包
                for packet in packets:
                    # 统计字节
                    packet_len = len(packet)
                    features["Total Bytes"] += packet_len

                    # 提取TCP标志
                    if TCP in packet:
                        tcp = packet[TCP]
                        if tcp.flags & 0x02:  # SYN
                            features["SYN Flag Count"] += 1
                        if tcp.flags & 0x04:  # RST
                            features["RST Flag Count"] += 1

                    # 确定方向并计数
                    if IP in packet:
                        if packet[IP].src == src_ip:
                            features["Total Fwd Packets"] += 1
                            payload_len = len(PcapAnalyzer._get_payload(packet))
                            if payload_len > features["Fwd Packet Length Max"]:
                                features["Fwd Packet Length Max"] = payload_len
                        else:
                            features["Total Bwd Packets"] += 1
                            payload_len = len(PcapAnalyzer._get_payload(packet))
                            if payload_len > features["Bwd Packet Length Max"]:
                                features["Bwd Packet Length Max"] = payload_len

                    # 收集原始数据（只取前10KB防止内存溢出）
                    if len(raw_data) < 10240:  # 10KB
                        raw_data += PcapAnalyzer._get_payload(packet)

                # 计算字节速率
                if flow_duration > 0:
                    features["Flow Bytes/s"] = features["Total Bytes"] / flow_duration

                # 构建告警对象
                alert_id = f"flow-{src_ip}:{sport}-{dst_ip}:{dport}-{proto}-{i}"
                try:
                    raw_text = raw_data.decode('utf-8', errors='ignore')
                except:
                    raw_text = "二进制数据无法解码"

                first_ts = float(first_pkt.time)
                last_ts = float(last_pkt.time)
                alert = {
                    "id": alert_id,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "sport": sport,
                    "dport": dport,
                    "proto": proto,
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(first_ts)),
                    "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts)),
                    "raw_data": raw_text,  # 直接包含在alert对象中
                    "features": features
                }

                alerts.append(alert)

        logger.info(f"生成 {len(alerts)} 个告警对象")
        return alerts


def main():
    pcap_file = 'sample.pcap'
    rule_dir = '../rules'

    # 初始化规则引擎
    engine = EnhancedRuleEngine(rule_dir)

    # 分析PCAP文件
    print(f"\n分析PCAP文件: {pcap_file}")
    alerts = PcapAnalyzer.analyze_pcap(pcap_file)
    print(f"发现 {len(alerts)} 个网络流")

    # 检测异常
    print("\n开始异常检测...")
    results = []
    for alert in alerts:
        matches = engine.match(alert)
        if matches:
            results.append({
                "alert": alert,
                "matches": matches
            })

    # 输出结果
    if not results:
        print("\n未检测到异常流量")
        return

    print(f"\n检测到 {len(results)} 个异常流:")
    for i, result in enumerate(results, 1):
        alert = result['alert']
        matches = result['matches']

        print(f"\n{'=' * 50}")
        print(f"异常 #{i}: {alert['id']}")
        print(f"{'-' * 50}")
        print(f"  协议: {'TCP' if alert['proto'] == 6 else 'UDP' if alert['proto'] == 17 else 'Other'}")
        print(f"  源地址: {alert['src_ip']}:{alert['sport']}")
        print(f"  目的地址: {alert['dst_ip']}:{alert['dport']}")
        print(f"  开始时间: {alert['start_time']}")
        print(f"  结束时间: {alert['end_time']}")
        print(f"  持续时间: {alert['features']['Flow Duration']:.2f}秒")

        print("\n  触发规则:")
        for match in matches:
            print(f"    - [{match['severity'].upper()}] {match['rule_name']} (ID: {match['rule_id']})")
            print(f"      描述: {match['description']}")
            print(f"      动作: {match.get('action', 'alert')}")

            # 显示条件详情
            print("\n      条件详情:")
            for cond_idx, cond_detail in enumerate(match['condition_details'], 1):
                status = "满足" if cond_detail['result'] else "不满足"
                value = cond_detail['value']
                if isinstance(value, list):
                    value = ", ".join(map(str, value))
                print(f"        条件{cond_idx}: 字段[{cond_detail['field']}] "
                      f"操作[{cond_detail['operator']}] "
                      f"值[{value}] -> {status}")

        # 显示关键特征
        print("\n  关键特征:")
        features = alert['features']
        for feature in ['SYN Flag Count', 'RST Flag Count',
                        'Total Fwd Packets', 'Total Bwd Packets',
                        'Fwd Packet Length Max', 'Bwd Packet Length Max',
                        'Total Bytes', 'Flow Bytes/s']:
            if feature in features:
                print(f"    - {feature}: {features[feature]}")

        # 显示原始数据摘要
        raw_data = alert['raw_data']
        if len(raw_data) > 200:
            print(f"\n  原始数据摘要: {raw_data[:200]}...")
        elif raw_data:
            print(f"\n  原始数据摘要: {raw_data}")

        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()