# rag_engine.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" #使用国内镜像

import logging
from typing import List, Optional
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_qdrant import Qdrant
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 禁用警告
logging.getLogger("langchain").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


class MedicalRAG:
    def __init__(self, data_path: str = "data/", collection_name: str = "medical_db"):
        self.data_path = data_path
        # === 新增：自动将 PDF 转为 TXT ===
        self._auto_convert_pdfs()
        self.collection_name = collection_name
        self._init_models()
        try:
            self._load_and_index_documents()
        except Exception as e:
            print(f"向量库构建失败: {e}")
            raise
        self._build_rag_chain()

    def _auto_convert_pdfs(self):
        """
        扫描 data_path 目录，将所有 .pdf 文件转换为 .txt（如果尚未转换）。
        转换后的文件保存在同一目录，文件名保持一致（仅扩展名变化）。
        """
        try:
            import os
            from pathlib import Path

            data_dir = Path(self.data_path)
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
                return

            pdf_files = list(data_dir.glob("*.pdf"))
            if not pdf_files:
                return  # 无 PDF，跳过

            # 检查哪些 PDF 尚未转换（即没有同名 .txt）
            to_convert = []
            for pdf in pdf_files:
                txt_path = pdf.with_suffix(".txt")
                if not txt_path.exists():
                    to_convert.append(pdf)

            if not to_convert:
                return  # 所有 PDF 已转换

            print(f"🔍 检测到 {len(to_convert)} 个未转换的 PDF 文件，正在自动转换为 TXT...")

            # 动态导入转换函数
            from pdf_to_txt_clean import convert_pdf_to_txt

            for pdf_path in to_convert:
                print(f"  → 转换: {pdf_path.name}")
                try:
                    convert_pdf_to_txt(str(pdf_path), output_dir=str(data_dir))
                except Exception as e:
                    print(f"转换失败 {pdf_path.name}: {e}")

            print("PDF 转 TXT 完成。\n")

        except Exception as e:
            print(f"PDF 自动转换过程出错: {e}")

    def _init_models(self):
        print("加载嵌入模型 (BAAI/bge-m3)...")
        try:
            # 尝试加载本地模型
            model_path = "models/bge-m3"
            if os.path.exists(model_path):
                print(f"→ 使用本地模型: {model_path}")
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            else:
                # 网络下载（备用）
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3",
                    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
        except Exception as e:
            print(f"嵌入模型加载失败: {str(e)}")
            print("→ 使用轻量级备用模型: sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )

        print("加载重排器 (BAAI/bge-reranker-v2-m3)...")
        try:
            reranker_path = "models/bge-reranker-v2-m3"
            if os.path.exists(reranker_path):
                print(f"→ 使用本地重排器模型: {reranker_path}")
                self.reranker = HuggingFaceCrossEncoder(model_name=reranker_path)
            else:
                self.reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        except Exception as e:
            print(f" 重排器加载失败: {str(e)}")
            print("→ 使用备用重排器: cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

        print("加载本地 LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
        try:
            llm_path = "models/qwen2.5-0.5b"
            if os.path.exists(llm_path):
                print(f"→ 使用本地LLM: {llm_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    llm_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
            else:
                model_name = "Qwen/Qwen2.5-0.5B-Instruct"
                print(f"→ 从Hugging Face下载LLM: {model_name} (这可能需要几分钟)")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=False,
                repetition_penalty=1.1
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"LLM加载失败: {str(e)}")
            print("→ 使用超轻量级模型: microsoft/Phi-3-mini-4k-instruct (需要在线下载)")
            try:
                from transformers import Phi3ForCausalLM, AutoTokenizer
                model = Phi3ForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct",
                    device_map="auto",
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
                tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.3
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
            except Exception as e2:
                print(f" 备用LLM加载失败: {str(e2)}")
                print(" 无法加载任何语言模型。请检查网络连接或创建 models/ 目录并放置本地模型。")
                raise

    def _load_and_index_documents(self):
        print("加载医学文档...")
        documents = []
        for file in os.listdir(self.data_path):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(self.data_path, file), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file
                documents.extend(docs)

        if not documents:
            raise ValueError("未找到任何医学文档！请检查 data/ 目录。")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", " "]
        )
        chunks = text_splitter.split_documents(documents)

        print(f"共生成 {len(chunks)} 个文本块，正在构建向量库...")
        # self.vector_store = Qdrant.from_documents(
        #     chunks,
        #     self.embedding_model,
        #     location=":memory:",
        #     collection_name=self.collection_name
        # )
        #  Qdrant ，改用磁盘持久化
        self.vector_store = Qdrant.from_documents(
            chunks,
            self.embedding_model,
            path="./qdrant_db",  # 持久化到本地目录
            collection_name=self.collection_name,
            force_recreate=True  # 避免重复覆盖（可选）
        )

        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        compressor = CrossEncoderReranker(model=self.reranker, top_n=3)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    def _rewrite_query(self, query: str) -> str:
        rewrite_rules = {
            "血糖高": "高血糖",
            "糖尿": "糖尿病",
            "怀孕血糖": "妊娠期糖尿病"
        }
        rewritten = query
        for k, v in rewrite_rules.items():
            rewritten = rewritten.replace(k, v)
        if rewritten != query:
            print(f"原始查询: {query} → 重写后: {rewritten}")
        return rewritten

    def _build_rag_chain(self):
        def format_docs_with_sources(docs):
            if not docs or all(not doc.page_content.strip() for doc in docs):
                return {"context": "", "sources": ""}
            context = "\n\n".join(doc.page_content for doc in docs)
            sources = sorted(set(doc.metadata.get("source", "未知来源") for doc in docs))
            return {"context": context, "sources": "、".join(sources)}

        prompt_template = """
    你是一名专业医疗助手，请严格依据以下医学资料回答问题。
    如果资料中无相关信息，请回答："根据当前知识库无法回答该问题"。

    【医学资料】
    {context}

    【问题】
    {question}

    【要求】
    1. 回答必须准确、简洁；
    2. 禁止任何猜测、编造或超出资料范围的建议。
    """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # 1. 打印原始查询 & 重写后查询
        def log_and_rewrite(question_str):
            print(f"原始查询: {question_str}")
            rewritten = self._rewrite_query(question_str)
            print(f"重写后查询: {rewritten}")
            return {"rewritten_q": rewritten, "original_q": question_str}

        input_mapper = RunnableLambda(log_and_rewrite)

        # 2. 打印重排后的文本（即 retriever 输出）
        def retrieve_and_log(x):
            docs = self.retriever.invoke(x["rewritten_q"])
            print("\n重排后返回的文本块（Top 3）:")
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()[:200].replace('\n', ' ')
                source = doc.metadata.get("source", "未知")
                print(f"  [{i}] 来源: {source} | 内容: {content}...")
            print()
            return {"docs": docs, "question": x["original_q"]}

        # 3. 打印最终注入提示词的上下文
        def prepare_prompt_input(x):
            info = format_docs_with_sources(x["docs"])
            context = info["context"]
            sources = info["sources"]
            question = x["question"]
            print("最终提示词（Prompt）输入内容:")
            final_prompt = prompt.format(context=context, question=question)
            print(final_prompt)
            print("\n" + "=" * 60 + "\n")
            return {
                "context": context,
                "sources": sources,
                "question": question
            }

        base_chain = (
                input_mapper
                | RunnableLambda(retrieve_and_log)
                | RunnableLambda(prepare_prompt_input)
                | {
                    "answer": prompt | self.llm | StrOutputParser(),
                    "sources": lambda x: x["sources"],
                }
        )

        def finalize_output(inputs):
            answer = inputs["answer"]
            sources = inputs["sources"]
            if not sources or "根据当前知识库无法回答" in answer:
                return answer
            return f"{answer}\n\n（资料来源：{sources}）"

        self.rag_chain = base_chain | RunnableLambda(finalize_output)
    def ask(self, question: str) -> str:
        return self.rag_chain.invoke(question)  # 直接传字符串