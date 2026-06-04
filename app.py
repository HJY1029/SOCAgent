# -*- coding: utf-8 -*-
"""
Web 后端：提供文件上传与 PCAP 分析 API，供前端调用。
运行方式: python app.py
前端访问: http://127.0.0.1:5000/
"""
import os
import sys
import json
import logging
import time
from datetime import datetime
from werkzeug.utils import secure_filename

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    print("请先安装 Flask: pip install flask")
    sys.exit(1)

# 上传目录与允许的扩展名
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {"pcap", "pcapng", "cap"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _exit_pause():
    """退出前暂停，防止终端自动关闭（便于查看输出）"""
    try:
        input("\n按回车键关闭...")
    except EOFError:
        # 无 TTY（如 IDE 运行、后台）时无法 input，改为等待数秒
        wait_sec = 30
        print(f"\n终端将在 {wait_sec} 秒后关闭，请查看上方输出。")
        try:
            time.sleep(wait_sec)
        except KeyboardInterrupt:
            pass
    except KeyboardInterrupt:
        pass


def _excepthook(typ, value, tb):
    """未捕获异常时打印并暂停，便于排查终端自动退出的问题"""
    sys.__excepthook__(typ, value, tb)
    _exit_pause()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.after_request
def add_cors_headers(response):
    """允许前端跨域访问（若前端与后端同源可忽略）"""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def index():
    """提供前端页面"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    """健康检查，便于确认后端是否启动"""
    return jsonify({"status": "ok", "message": "backend is running"})


@app.route("/api/upload", methods=["POST", "OPTIONS"])
def upload_and_analyze():
    """
    接收前端上传的 PCAP 文件，调用 agent 进行分析，返回报告 JSON。
    前端需以 multipart/form-data 提交字段名: file
    """
    if request.method == "OPTIONS":
        return "", 204
    if "file" not in request.files:
        return jsonify({"success": False, "error": "未找到上传字段 'file'"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"success": False, "error": "未选择文件"}), 400

    if not allowed_file(f.filename):
        return jsonify({
            "success": False,
            "error": "仅支持 .pcap / .pcapng / .cap 文件"
        }), 400

    try:
        filename = secure_filename(f.filename)
        # 避免重名覆盖：加时间戳
        base, ext = os.path.splitext(filename)
        safe_name = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        f.save(save_path)
        logger.info("文件已保存: %s", save_path)
    except Exception as e:
        logger.exception("保存上传文件失败")
        return jsonify({"success": False, "error": f"保存文件失败: {str(e)}"}), 500

    # 调用分析引擎（延迟导入，避免启动时加载大模型）
    try:
        from agent import AlertAgent

        config = {
            "knowledge_dir": "knowledge_base",
            "rule_dir": "rules",
            "use_llm": False,  # 可改为 True 启用 LLM，需配置模型路径
            "llm_model_path": "llm_models/deepseek-llm-r1",
        }
        agent = AlertAgent(config)
        report = agent.analyze_pcap(save_path)
        try:
            agent.save_report(report)
        except Exception as e:
            logger.warning("保存报告到文件失败（仍返回报告）: %s", e)
        return jsonify({"success": True, "report": report})
    except SystemExit as e:
        # 防止依赖库内部 sys.exit() 导致整个进程退出
        logger.exception("分析过程中发生 SystemExit，已拦截")
        return jsonify({
            "success": False,
            "error": "分析过程异常退出",
            "detail": str(e),
        }), 500
    except Exception as e:
        logger.exception("分析过程出错")
        return jsonify({
            "success": False,
            "error": f"分析失败: {str(e)}",
            "detail": str(e),
        }), 500


if __name__ == "__main__":
    sys.excepthook = _excepthook
    use_reloader = os.environ.get("FLASK_RELOAD", "").lower() in ("1", "true", "yes")
    print("Backend: http://127.0.0.1:5000/")
    print("API:    POST http://127.0.0.1:5000/api/upload (field: file)")
    if use_reloader:
        print("Hot reload: 已开启 (FLASK_RELOAD=1)")
    else:
        print("Hot reload: 未开启（分析后进程不会退出；需热重载请设 FLASK_RELOAD=1）")
    print("退出服务后需按回车键才会关闭终端（防止窗口自动关）")
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=use_reloader)
    except Exception:
        logger.exception("服务异常退出")
        raise
    finally:
        _exit_pause()
