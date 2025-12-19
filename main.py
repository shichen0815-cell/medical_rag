# main.py
import logging
import os
import sys
from pathlib import Path

if os.path.exists("./qdrant_db/qdrant.lock"):
    os.remove("./qdrant_db/qdrant.lock")


def setup_logging(log_file="logs/medical_rag.log", level=logging.DEBUG):
    """统一日志配置：同时输出到控制台和文件"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def safe_input(prompt: str) -> str:
    sys.stdout.write("\n")
    sys.stdout.flush()
    return input(prompt)


def load_docs_text(docs_dir: str) -> str:
    """读取 docs 目录下所有 txt 病历文件"""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"{docs_dir} 目录不存在")

    texts = []
    for file in sorted(docs_path.glob("*.txt")):
        logging.info(f"读取病历文件: {file.name}")
        with open(file, "r", encoding="utf-8") as f:
            texts.append(f"\n===== {file.name} =====\n")
            texts.append(f.read())

    if not texts:
        raise ValueError("docs 目录下未找到任何 txt 病历文件")

    return "\n".join(texts)


def select_mode() -> str:
    """模式选择：1=病历检阅，2=问答"""
    while True:
        print("\n请选择运行模式：")
        print("1. 病历检阅（读取 docs/ 目录）")
        print("2. 医疗问答（交互式提问）")
        choice = safe_input("请输入 1 或 2: ").strip()

        if choice in ("1", "2"):
            return choice

        logging.warning("无效输入，请输入 1 或 2")


def run_review_mode(rag):
    """病历检阅模式"""
    logging.info("进入病历检阅模式")
    try:
        medical_text = load_docs_text("docs/medical_record")
        logging.info("病历读取完成，开始 RAG 检阅...medical_text: %s", medical_text)
        result = rag.review_record(medical_text)
        logging.info("病历检阅结果：\n%s", result)
    except Exception as e:
        logging.error(f"病历检阅失败: {e}", exc_info=True)


def run_qa_mode(rag):
    """问答模式"""
    logging.info("进入医疗问答模式（输入 quit 退出）")
    while True:
        question = safe_input("您的问题: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            logging.info("退出问答模式")
            break

        if not question:
            continue

        try:
            answer = rag.ask(question)
            logging.info("回答:\n%s", answer)
        except Exception as e:
            logging.error(f"回答生成出错: {e}", exc_info=True)


def main():
    logging.info("正在初始化医疗 RAG 系统...")
    try:
        from rag_engine import MedicalRAG
        rag = MedicalRAG(data_path="data/")

        mode = select_mode()

        if mode == "1":
            run_review_mode(rag)
        else:
            run_qa_mode(rag)

    except Exception as e:
        logging.critical(f"系统初始化失败: {e}", exc_info=True)
        logging.info("请检查：")
        logging.info("1. data/ 目录下是否有医学文档")
        logging.info("2. docs/ 目录是否存在（病历检阅模式）")
        logging.info("3. 依赖是否正确安装")


if __name__ == "__main__":
    setup_logging()
    main()
