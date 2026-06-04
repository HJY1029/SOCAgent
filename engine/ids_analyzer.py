import os
import requests
import json
import re
# 在文件顶部添加以下导入语句
import dpkt
import socket

from datetime import datetime
from collections import defaultdict

# 配置参数
SNORT_RULES_URL = "https://rules.emergingthreats.net/open/snort-3.0/rules/"


def _get_rules_dir():
    """始终基于当前文件位置解析 rules 目录，避免硬编码或缓存问题"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")


RULES_DIR = _get_rules_dir()  # 兼容其他引用
PCAP_FILE = "sample.pcap"
OUTPUT_FILE = "alerts.json"

# 规则分类和优先级映射
RULE_CLASSIFICATIONS = {
    "not-suspicious": ("Not Suspicious Traffic", 3),
    "bad-unknown": ("Potentially Bad Traffic", 2),
    "attempted-recon": ("Attempted Information Leak", 2),
    "successful-recon": ("Information Leak", 2),
    "attempted-dos": ("Attempted Denial of Service", 2),
    "successful-dos": ("Denial of Service", 2),
    "attempted-user": ("Attempted User Privilege Gain", 1),
    "successful-user": ("User Privilege Gain", 1),
    "attempted-admin": ("Attempted Administrator Privilege Gain", 1),
    "successful-admin": ("Administrator Privilege Gain", 1),
}


class RuleEngine:
    def __init__(self):
        self.rules = []
        self.content_rules = defaultdict(list)
        self.flow_rules = defaultdict(list)
        self.signature_rules = defaultdict(list)
        self.sid_counter = 1000000  # 为自定义规则分配SID

    def add_rule(self, rule_line):
        """解析并添加Snort规则"""
        try:
            header_match = re.match(
                r'^(alert|log)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\((.*)\)',
                rule_line
            )
            if not header_match:
                return False

            groups = header_match.groups()
            action, protocol, src_ip, src_port, direction, dst_ip, dst_port = groups[:7]
            options_str = groups[7]

            # 改进的选项解析器
            option_dict = {}
            remaining = options_str.strip()

            while remaining:
                # 尝试匹配带引号的值
                quoted_match = re.match(r'(\w+):\s*"((?:[^"\\]|\\.)*)"\s*(;|$)', remaining)
                if quoted_match:
                    key, value, terminator = quoted_match.groups()
                    option_dict[key.strip()] = value.strip()
                    remaining = remaining[quoted_match.end():].strip()
                    continue

                # 尝试匹配不带引号的值
                unquoted_match = re.match(r'(\w+):\s*([^;]+)\s*(;|$)', remaining)
                if unquoted_match:
                    key, value, terminator = unquoted_match.groups()
                    option_dict[key.strip()] = value.strip()
                    remaining = remaining[unquoted_match.end():].strip()
                    continue

                # 尝试匹配标志选项（无值）
                flag_match = re.match(r'(\w+)\s*(;|$)', remaining)
                if flag_match:
                    key, terminator = flag_match.groups()
                    option_dict[key.strip()] = ""
                    remaining = remaining[flag_match.end():].strip()
                    continue

                break  # 无法解析更多选项

            # 创建规则对象
            rule = {
                "raw": rule_line,
                "action": action,
                "protocol": protocol,
                "src_ip": src_ip,
                "src_port": src_port,
                "direction": direction,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "options": option_dict,
                "sid": int(option_dict.get("sid", self.sid_counter)),
                "msg": option_dict.get("msg", "No message"),
                "classtype": option_dict.get("classtype", "unknown"),
                "priority": int(option_dict.get("priority", 3)),
                "content": option_dict.get("content", None),
                "flow": option_dict.get("flow", None),
                "pcre": option_dict.get("pcre", None),
            }

            self.sid_counter += 1
            self.rules.append(rule)

            # 索引规则以便快速查找
            if rule["content"]:
                # 提取内容模式（简化处理）
                content_pattern = re.search(r'!"?([^";]+)"?', rule["content"])
                if content_pattern:
                    pattern = content_pattern.group(1)
                    self.content_rules[rule["protocol"]].append((pattern, rule))

            if rule["flow"]:
                self.flow_rules[rule["protocol"]].append(rule)

            # 为所有规则创建签名索引
            key = f"{rule['protocol']}_{rule['src_port']}_{rule['dst_port']}"
            self.signature_rules[key].append(rule)

            return True
        except Exception as e:
            print(f"解析规则失败: {rule_line} - {e}")
            return False

    def match_packet(self, packet):
        """检查数据包是否匹配任何规则"""
        matches = []

        # 提取数据包基本信息
        try:
            if isinstance(packet.data, dpkt.ip.IP):
                ip = packet.data
                protocol = ip.p
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)

                # 提取端口信息
                src_port, dst_port = 0, 0
                if protocol == dpkt.ip.IP_PROTO_TCP:
                    tcp = ip.data
                    src_port = tcp.sport
                    dst_port = tcp.dport
                    payload = tcp.data
                elif protocol == dpkt.ip.IP_PROTO_UDP:
                    udp = ip.data
                    src_port = udp.sport
                    dst_port = udp.dport
                    payload = udp.data
                else:
                    payload = ip.data

                # 转换为协议名称
                protocol_name = {
                    dpkt.ip.IP_PROTO_TCP: "tcp",
                    dpkt.ip.IP_PROTO_UDP: "udp",
                    dpkt.ip.IP_PROTO_ICMP: "icmp"
                }.get(protocol, str(protocol))

                # 检查基于签名的规则
                sig_key = f"{protocol_name}_{src_port}_{dst_port}"
                for rule in self.signature_rules.get(sig_key, []):
                    # 检查IP匹配
                    if not self._match_ip(src_ip, rule["src_ip"]):
                        continue
                    if not self._match_ip(dst_ip, rule["dst_ip"]):
                        continue

                    # 检查端口匹配
                    if not self._match_port(src_port, rule["src_port"]):
                        continue
                    if not self._match_port(dst_port, rule["dst_port"]):
                        continue

                    # 检查内容匹配
                    if rule["content"] and rule["content"] not in str(payload):
                        continue

                    # 检查流状态
                    if rule["flow"] and not self._match_flow(rule["flow"]):
                        continue

                    # 匹配成功
                    matches.append(rule)

                # 检查基于内容的规则
                for pattern, rule in self.content_rules.get(protocol_name, []):
                    if pattern in str(payload):
                        matches.append(rule)

            return matches
        except Exception as e:
            print(f"处理数据包时出错: {e}")
            return []

    def _match_ip(self, ip, rule_ip):
        """检查IP是否匹配规则"""
        if rule_ip == "any":
            return True
        if rule_ip.startswith("!") and ip != rule_ip[1:]:
            return True
        if ip == rule_ip:
            return True
        if "/" in rule_ip:
            # 简化处理 - 实际应实现CIDR匹配
            return ip.startswith(rule_ip.split("/")[0])
        return False

    def _match_port(self, port, rule_port):
        """检查端口是否匹配规则"""
        if rule_port == "any":
            return True
        if rule_port.startswith("!") and port != int(rule_port[1:]):
            return True
        if ":" in rule_port:
            start, end = map(int, rule_port.split(":"))
            return start <= port <= end
        try:
            return port == int(rule_port)
        except:
            return False

    def _match_flow(self, flow):
        """检查流状态（简化实现）"""
        # 实际应维护流状态，这里返回True表示匹配
        return True


def download_rules():
    """下载Snort规则"""
    rules_dir = _get_rules_dir()
    os.makedirs(rules_dir, exist_ok=True)

    # 下载Snort规则
    print("正在下载Snort规则...")
    snort_rules = [
        "emerging-exploit.rules",
        "emerging-malware.rules",
        "emerging-attack_response.rules",
        "emerging-shellcode.rules",
        "emerging-web_client.rules",
        "emerging-web_server.rules"
    ]

    for rule_file in snort_rules:
        print(f"加载规则文件: {rule_file}")
        url = f"{SNORT_RULES_URL}{rule_file}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(os.path.join(rules_dir, rule_file), "wb") as f:
                f.write(response.content)
            print(f"下载成功: {rule_file}")
        except Exception as e:
            print(f"下载失败 {rule_file}: {e}")

    print("Snort规则下载完成")


def load_rules(engine):
    """加载所有规则到引擎"""
    rules_dir = _get_rules_dir()
    if not os.path.isdir(rules_dir):
        os.makedirs(rules_dir, exist_ok=True)
        download_rules()
    rule_files = [f for f in os.listdir(rules_dir) if f.endswith(".rules")]
    total_rules = 0

    for rule_file in rule_files:
        with open(os.path.join(rules_dir, rule_file), "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if engine.add_rule(line):
                        total_rules += 1

    print(f"已加载 {total_rules} 条规则到引擎")
    return engine


def analyze_pcap(pcap_file, engine):
    """分析PCAP文件并检测攻击"""
    alerts = []

    try:
        with open(pcap_file, "rb") as f:
            pcap = dpkt.pcap.Reader(f)

            for timestamp, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue

                    matches = engine.match_packet(eth)
                    for rule in matches:
                        ip = eth.data
                        src_ip = socket.inet_ntoa(ip.src)
                        dst_ip = socket.inet_ntoa(ip.dst)

                        alert = {
                            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
                            "src_ip": src_ip,
                            "dst_ip": dst_ip,
                            "src_port": 0,
                            "dst_port": 0,
                            "protocol": rule["protocol"],
                            "msg": rule["msg"],
                            "sid": rule["sid"],
                            "classtype": rule["classtype"],
                            "priority": rule["priority"],
                            "classification": RULE_CLASSIFICATIONS.get(
                                rule["classtype"],
                                ("Unknown", 3)
                            )[0],
                            "severity": RULE_CLASSIFICATIONS.get(
                                rule["classtype"],
                                ("Unknown", 3)
                            )[1],
                            "rule_content": rule["raw"]
                        }

                        # 尝试获取端口信息
                        if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                            alert["src_port"] = ip.data.sport
                            alert["dst_port"] = ip.data.dport

                        alerts.append(alert)
                        print(f"检测到警报: {rule['msg']} (SID: {rule['sid']})")

                except Exception as e:
                    print(f"处理数据包时出错: {e}")

    except Exception as e:
        print(f"读取PCAP文件失败: {e}")

    print(f"共检测到 {len(alerts)} 个安全事件")
    return alerts


def save_results(alerts, output_file):
    """保存检测结果"""
    try:
        with open(output_file, "w") as f:
            json.dump(alerts, f, indent=2)
        print(f"结果已保存至 {output_file}")
    except Exception as e:
        print(f"保存结果失败: {e}")


def main():
    # 1. 下载规则
    download_rules()

    # 2. 初始化规则引擎
    engine = RuleEngine()
    load_rules(engine)

    # 3. 检查PCAP文件是否存在
    if not os.path.exists(PCAP_FILE):
        print(f"错误: PCAP文件 {PCAP_FILE} 不存在")
        return

    # 4. 分析PCAP文件
    alerts = analyze_pcap(PCAP_FILE, engine)

    # 5. 保存结果
    save_results(alerts, OUTPUT_FILE)

    print("分析完成")


if __name__ == "__main__":
    main()