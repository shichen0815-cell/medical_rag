import os
import glob
import logging
from graph_builder import MedicalGraphManager

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("InitGraph")


def init_all_data(data_dir="data"):
    manager = MedicalGraphManager()

    if not manager.driver:
        logger.error("无法连接图数据库，终止初始化。")
        return

    # 获取所有 txt 文件
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    logger.info(f"在 {data_dir} 下发现 {len(txt_files)} 个说明书文件。")

    success_count = 0

    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 调用新增接口
            logger.info(f"正在处理: {file_name} ...")
            if manager.add_document(content, file_name):
                success_count += 1

        except Exception as e:
            logger.error(f"处理文件 {file_name} 时出错: {e}")

    logger.info(f"初始化完成。成功导入 {success_count}/{len(txt_files)} 个文件。")
    manager.close()


if __name__ == "__main__":
    # 请确保你的 txt 文件都在 data/ 目录下
    init_all_data("data")
    #python init_graph_db.py 读取data下的文件，更新到图数据库。