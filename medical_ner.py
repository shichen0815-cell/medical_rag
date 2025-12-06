# medical_ner.py
import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict

logger = logging.getLogger(__name__)


class NER:
    """
    中文医学命名实体识别（基于 lixin12345/chinese-medical-ner）
    支持实体类型：
      - Drug
      - DiseaseNameOrComprehensiveCertificate
      - TreatmentOrPreventionProcedures
    """

    def __init__(self, model_path: str = "lixin12345/chinese-medical-ner"):
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"医学 NER 模型 '{model_path}' 加载成功（设备: {self.device}）。")
        except Exception as e:
            raise RuntimeError(f"加载 NER 模型失败: {e}")

    def ner(self, sentence: str) -> List[Dict]:
        """
        识别句子中的医学实体（支持长文本自动分段）。
        Args:
            sentence (str): 输入文本
        Returns:
            List[Dict]: 实体列表，每个 dict 包含 'type' 和 'tokens'
        """
        if not sentence or not isinstance(sentence, str):
            return []
        ans = []
        for i in range(0, len(sentence), 500):
            segment = sentence[i:i + 500]
            ans.extend(self._ner(segment))
        return ans

    def _ner(self, sentence: str) -> List[Dict]:
        if len(sentence) == 0:
            return []

        inputs = self.tokenizer(
            sentence,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = logits.argmax(-1)
        predicted_labels = [self.model.config.id2label[t.item()] for t in predicted_ids[0]]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = []
        current_entity = None

        for idx, (token, label) in enumerate(zip(tokens, predicted_labels)):
            if token in {"[CLS]", "[SEP]", "[PAD]"}:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            clean_token = token[2:] if token.startswith("##") else token

            if label.startswith(("B-", "S-")):
                if current_entity:
                    entities.append(current_entity)
                ent_type = label.replace("B-", "").replace("S-", "")
                current_entity = {"type": ent_type, "tokens": [clean_token]}
            elif label.startswith(("I-", "E-", "M-")):
                if current_entity is None:
                    ent_type = label.replace("I-", "").replace("E-", "").replace("M-", "")
                    current_entity = {"type": ent_type, "tokens": [clean_token]}
                else:
                    current_entity["tokens"].append(clean_token)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities