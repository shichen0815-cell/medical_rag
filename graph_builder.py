import os
import json
import logging
import re
from typing import Dict, Any, List
from neo4j import GraphDatabase

# 复用你的模型工厂（确保 model_factory.py 在同一目录）
from model_factory import ModelFactory

logger = logging.getLogger(__name__)


class MedicalGraphManager:
    def __init__(self, uri=None, user=None, password=None):
        # 1. 初始化 Neo4j 连接
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://192.168.43.225:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "Neo4j9527")

        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.verify_connection()
            self.init_schema()  # 启动时自动建立索引
        except Exception as e:
            logger.error(f"图数据库连接失败: {e}")

        # 2. 初始化 LLM (用于提取实体)
        self.models = ModelFactory()

    def close(self):
        if self.driver:
            self.driver.close()

    def verify_connection(self):
        with self.driver.session() as session:
            session.run("RETURN 1")
        logger.info("✅ Neo4j 连接成功")

    def init_schema(self):
        """初始化图谱的约束和索引，防止重复数据"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Population) REQUIRE n.name IS UNIQUE",  # 人群：如 4岁、孕妇
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Component) REQUIRE n.name IS UNIQUE"
        ]
        try:
            with self.driver.session() as session:
                for c in constraints:
                    session.run(c)
            logger.info("图谱 Schema 约束已就绪")
        except Exception as e:
            logger.warning(f"Schema 初始化警告: {e}")

    def _extract_struct_from_text(self, text: str) -> Dict[str, Any]:
        """
        核心方法：调用 LLM 将文本说明书转化为结构化 JSON
        """
        prompt = f"""
        你是一个专业的医学知识图谱构建专家。请阅读以下【药品说明书】，提取关键实体和关系，输出严格的 JSON 格式。

        【提取要求】
        1. **name**: 药品通用名称（不包含商品名）。
        2. **usage_dosage**: 提取用法用量的核心文本。
        3. **indications**: (List) 适应症、治疗的疾病列表。
        4. **contraindications**: (List) **关键**！提取所有禁忌对象。
           - 必须包含具体的疾病（如“严重肾功能不全”）。
           - **必须包含人群限制**（如“18岁以下”、“孕妇”、“哺乳期”、“新生儿”）。
           - 如果原文说“4岁以下禁用”，请提取“4岁以下”作为实体。
        5. **components**: (List) 主要成分列表。
        6. **adverse_reactions**: (List) 常见不良反应症状。

        【重要：数据清洗】
        - 列表中的实体必须简洁，例如输出 ["头痛", "发热"] 而不是 ["缓解头痛", "治疗发热"]。
        - 遇到“禁用”字眼，务必将主语提取到 contraindications 中。

        【待处理文本】
        {text[:3000]} (文本截断以防超长)

        【输出格式】
        请仅输出 JSON 字符串，不要包含 Markdown 标记。
        {{
            "name": "...",
            "usage_dosage": "...",
            "indications": ["..."],
            "contraindications": ["..."],
            "components": ["..."],
            "adverse_reactions": ["..."]
        }}
        """

        try:
            # 调用你的 Ollama 模型
            response = self.models.ollama.generate([
                {"role": "system", "content": "你是一个严谨的图谱构建助手，只输出 JSON。"},
                {"role": "user", "content": prompt}
            ])

            # 清洗并解析 JSON
            cleaned_json = response.strip()
            if cleaned_json.startswith("```"):
                cleaned_json = re.sub(r"^```(json)?|```$", "", cleaned_json, flags=re.MULTILINE).strip()

            return json.loads(cleaned_json)
        except Exception as e:
            logger.error(f"LLM 结构化提取失败: {e}")
            return {}

    def add_document(self, text: str, source_filename: str):
        """
        【新增接口】解析文本并存入图数据库
        """
        logger.info(f"开始处理文档: {source_filename}")

        # 1. LLM 提取
        data = self._extract_struct_from_text(text)
        if not data or not data.get("name"):
            logger.error(f"无法从 {source_filename} 提取有效药品信息")
            return False

        # 2. 注入元数据
        data["source_file"] = source_filename

        # 3. Cypher 写入
        self._write_to_neo4j(data)
        return True

    def delete_document(self, drug_name: str):
        """
        【删除接口】根据药品名称删除相关节点及其关系
        注意：我们只删除 Drug 节点及其发出的关系，保留 Disease 等公用节点
        """
        cypher = """
        MATCH (d:Drug {name: $name})
        DETACH DELETE d
        """
        try:
            with self.driver.session() as session:
                session.run(cypher, name=drug_name)
            logger.info(f"已从图谱中移除药品: {drug_name}")
            return True
        except Exception as e:
            logger.error(f"删除失败: {e}")
            return False

    def _write_to_neo4j(self, data: Dict[str, Any]):
        """执行 Cypher 写入"""
        # 确保数据里有默认空列表防止报错
        data.setdefault("indications", [])
        data.setdefault("contraindications", [])
        data.setdefault("components", [])
        data.setdefault("adverse_reactions", [])

        cypher = """
        MERGE (d:Drug {name: $name})
        SET d.usage_dosage = $usage_dosage,
            d.source_file = $source_file,
            d.update_time = datetime()

        // 1. 建立适应症关系 (Drug)-[:TREATING]->(Disease)
        FOREACH (ind IN $indications | 
            MERGE (t:Disease {name: ind})
            MERGE (d)-[:TREATING]->(t)
        )

        // 2. 建立禁忌关系 (Drug)-[:CONTRAINDICATES]->(Population/Target)
        // 这里我们将所有禁忌对象统一标记为 Population 或 Target，方便统一检索
        FOREACH (con IN $contraindications | 
            MERGE (p:Population {name: con}) 
            MERGE (d)-[:CONTRAINDICATES]->(p)
        )

        // 3. 建立成分关系
        FOREACH (comp IN $components | 
            MERGE (c:Component {name: comp})
            MERGE (d)-[:HAS_COMPONENT]->(c)
        )

        // 4. 建立不良反应关系
        FOREACH (adr IN $adverse_reactions | 
            MERGE (s:Symptom {name: adr})
            MERGE (d)-[:HAS_ADVERSE_REACTION]->(s)
        )
        """

        try:
            with self.driver.session() as session:
                session.run(cypher, **data)
            logger.info(f"图谱写入成功: {data['name']} (含 {len(data['contraindications'])} 条禁忌)")
        except Exception as e:
            logger.error(f"Cypher 写入异常: {e}")
            raise