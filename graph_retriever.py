# graph_retriever.py
import logging
import os
from typing import List, Set, Optional
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# === Neo4j 支持 ===
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("未安装 neo4j 包，图数据库功能将被禁用。请运行: pip install neo4j")

# === 医学 NER 支持 ===
try:
    from medical_ner import NER
    NER_AVAILABLE = True
except ImportError as e:
    NER_AVAILABLE = False
    logger.warning(f"医学 NER 模块不可用（{e}），图检索将跳过实体识别。")


class GraphRetriever:
    def __init__(self):
        self.driver = None
        self.ner_model: Optional[NER] = None

        # 初始化 Neo4j
        if not NEO4J_AVAILABLE:
            logger.info("Neo4j 依赖缺失，图检索器未启用。")
            return

        uri = os.getenv("NEO4J_URI", "bolt://192.168.43.225:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "Neo4j9527")

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j 图数据库连接成功。")
        except Exception as e:
            logger.warning(f"Neo4j 连接失败，图检索器将返回空结果: {e}")
            if self.driver:
                self.driver.close()
            self.driver = None

        # 初始化 NER 模型（仅当 Neo4j 可用时加载）
        if self.driver and NER_AVAILABLE:
            try:
                # 【修改点 1】: 获取当前脚本所在的绝对路径，确保路径准确
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # 假设 models 文件夹在项目根目录 (与 graph_retriever.py 同级或上一级)
                # 如果 graph_retriever.py 在根目录：
                target_model_path = os.path.join(current_dir, "models", "chinese-medical-ner")

                # 【修改点 2】: 检查该路径是否存在
                if os.path.exists(target_model_path):
                    logger.info(f"✅ 正在加载本地 NER 模型: {target_model_path}")
                    # 直接加载解压后的模型文件
                    self.ner_model = NER(target_model_path)
                else:
                    logger.warning(f"❌ 本地路径未找到: {target_model_path}")
                    logger.warning("⚠️ 正在尝试联网加载 (这将导致启动非常慢)...")
                    # 只有本地完全没有时，才使用 ID 加载
                    self.ner_model = NER("lixin12345/chinese-medical-ner")
            except Exception as e:
                logger.error(f"医学 NER 模型初始化失败: {e}")
                self.ner_model = None

    def _extract_medical_entities(self, query: str) -> List[str]:
        """
        提取 Drug 和 Disease 实体，并拼接为完整词。
        """
        if not self.ner_model or not query:
            return []

        try:
            raw_entities = self.ner_model.ner(query)
            keywords: Set[str] = set()

            for ent in raw_entities:
                ent_type = ent.get("type", "")
                tokens = ent.get("tokens", [])
                if not tokens:
                    continue

                entity_text = "".join(tokens).strip()
                # 只保留核心医学实体
                if ent_type in {"Drug", "DiseaseNameOrComprehensiveCertificate"} and len(entity_text) >= 2:
                    keywords.add(entity_text)

            result = list(keywords)
            logger.debug(f"NER 提取医学实体: {result}")
            return result

        except Exception as e:
            logger.error(f"医学实体提取失败: {e}")
            return []

    def retrieve(self, query: str) -> List[Document]:
        """
        [优化版] 从 Neo4j 检索，适配 MedicalGraphManager 的 Schema
        """
        if not self.driver:
            return []

        # 1. 提取实体 (NER + 简单的包含匹配作为兜底)
        keywords = self._extract_medical_entities(query)

        # 【兜底策略】如果 NER 没提取到，尝试直接把 query 当作关键词去数据库碰运气
        # 实际生产中，建议有一个药名词典做 AC 自动机匹配
        if not keywords and len(query) < 10:
            keywords = [query]

        if not keywords:
            logger.debug("未提取到有效实体，跳过图检索。")
            return []

        docs = []

        # 2. 定义关系映射表 (Schema -> 自然语言)
        # 对应 MedicalGraphManager 中的关系定义
        rel_map = {
            "TREATING": "治疗/适应症",
            "CONTRAINDICATES": "禁忌/禁止",
            "HAS_COMPONENT": "主要成分",
            "HAS_ADVERSE_REACTION": "不良反应/副作用"
        }

        try:
            with self.driver.session() as session:
                # === 查询 A: 获取药品节点的自身属性 (用法用量) ===
                # 你的写入逻辑中：d.usage_dosage
                usage_cypher = """
                MATCH (n:Drug)
                WHERE n.name IN $keywords
                RETURN n.name AS name, n.usage_dosage AS usage, n.source_file AS source
                """
                usage_result = session.run(usage_cypher, keywords=keywords)
                for record in usage_result:
                    name = record["name"]
                    usage = record["usage"]
                    source = record.get("source", "知识图谱")

                    if usage:
                        text = f"【用法用量】{name}的使用说明：{usage}"
                        docs.append(Document(page_content=text, metadata={"source": source, "type": "usage"}))

                # === 查询 B: 获取关联关系 ===
                # 修改点：不再查 r.description，而是查 type(r) 并手动映射
                rel_cypher = """
                MATCH (n)-[r]->(m)
                WHERE n.name IN $keywords
                RETURN n.name AS source, type(r) AS rel_type, m.name AS target
                LIMIT 20
                """
                rel_result = session.run(rel_cypher, keywords=keywords)

                # 聚合结果，避免碎片化 (比如把所有禁忌合并成一句话)
                relations_buffer = {
                    "TREATING": [],
                    "CONTRAINDICATES": [],
                    "HAS_COMPONENT": [],
                    "HAS_ADVERSE_REACTION": []
                }

                drug_name = ""
                for record in rel_result:
                    src = record["source"]
                    rel_type = record["rel_type"]
                    tgt = record["target"]
                    drug_name = src  # 记录当前药名

                    if rel_type in relations_buffer:
                        relations_buffer[rel_type].append(tgt)

                # 将聚合后的数据转换为自然语言 Document
                for r_type, targets in relations_buffer.items():
                    if not targets:
                        continue

                    cn_rel_name = rel_map.get(r_type, r_type)
                    targets_str = "、".join(targets)

                    # 生成符合人类阅读习惯的句子
                    if r_type == "CONTRAINDICATES":
                        content = f"【禁忌症】{drug_name}的{cn_rel_name}对象包括：{targets_str}。此类人群请禁用。"
                    elif r_type == "TREATING":
                        content = f"【适应症】{drug_name}主要用于{cn_rel_name}：{targets_str}。"
                    elif r_type == "HAS_ADVERSE_REACTION":
                        content = f"【不良反应】{drug_name}可能引起{cn_rel_name}：{targets_str}。"
                    else:
                        content = f"{drug_name}的{cn_rel_name}包括：{targets_str}。"

                    docs.append(Document(page_content=content, metadata={"source": "知识图谱", "type": "relation"}))

            logger.info(f"图数据库检索命中关键词 {keywords}，生成 {len(docs)} 条证据。")
            return docs

        except Exception as e:
            logger.error(f"图数据库查询出错: {e}")
            return []

    def query_relations(self, subject: str, object_: str = None) -> List[dict]:
        """
        模拟图数据库查询
        返回结构化数据: [{'rel': '禁忌', 'tail': '痛风', 'desc': '可能导致尿酸升高'}]
        """
        # 实际代码这里应该是 Neo4j / NetworkX 的查询逻辑
        # 这里做 Mock 数据演示逻辑
        results = []
        if "阿莫西林" in subject and "痛风" in (object_ or ""):
            results.append({
                "head": "阿莫西林",
                "relation": "慎用/禁忌",
                "tail": "痛风",
                "description": "阿莫西林可能干扰尿酸排泄，痛风患者需调整剂量或监测。",
                "source": "中国药典2020版-图谱库"
            })
        return results

    def __del__(self):
        if self.driver:
            self.driver.close()