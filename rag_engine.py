import os
from pathlib import Path
from typing import Optional
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from config.settings import Settings as AppSettings

class RAGEngine:
    """LlamaIndexを使用した完全なRAGシステム（検索 + 生成）"""
    
    def __init__(self):
        """RAGエンジンを初期化"""
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self.node_parser: Optional[SentenceSplitter] = None
        self._setup_llama_index()
    
    def _setup_llama_index(self):
        """LlamaIndexの設定を初期化（RAG専用のLLM設定）"""
        # RAG用LLM設定 - OpenRouter経由
        Settings.llm = OpenAI(
            api_key=AppSettings.OPENROUTER_API_KEY,
            api_base=AppSettings.OPENROUTER_BASE_URL,
            model="gpt-4o-mini",  # OpenRouter経由でもOpenAI形式のモデル名を使用
            temperature=0.2,  # RAGでは事実ベースなので低い温度
            system_prompt="あなたは提供されたドキュメントの内容に基づいて、正確で詳細な回答を提供するAIアシスタントです。ドキュメントに書かれていない情報については推測せず、明確に「ドキュメントには記載されていません」と伝えてください。"
        )
        
        # OpenAI Embeddingsの設定（OpenRouter経由）
        Settings.embed_model = OpenAIEmbedding(
            api_key=AppSettings.OPENROUTER_API_KEY,
            api_base=AppSettings.OPENROUTER_BASE_URL,
            model="text-embedding-3-small"
        )
        
        # チャンク分割の設定（日記に最適化）
        self.node_parser = SentenceSplitter(
            chunk_size=512,  # 日記の段落程度のサイズ
            chunk_overlap=50,  # 文脈保持のためのオーバーラップ
            separator="\n\n",  # 段落区切りを優先
        )
        Settings.node_parser = self.node_parser
    
    def load_documents(self) -> bool:
        """日記ファイルを読み込んでインデックスを構築（メタデータ付き）"""
        try:
            print("📖 Obsidianファイルを読み込み中...")
            
            # Markdownファイルのみを読み込み
            reader = SimpleDirectoryReader(
                input_dir=AppSettings.DIARY_PATH,
                required_exts=[".md"],
                recursive=True
            )
            
            raw_documents = reader.load_data()
            
            if not raw_documents:
                print(f"警告: {AppSettings.DIARY_PATH} にMarkdownファイルが見つかりません")
                return False
            
            print(f"📄 読み込んだファイル数: {len(raw_documents)}")
            
            # メタデータを追加してドキュメントを強化
            enhanced_documents = []
            for doc in raw_documents:
                # ファイル名から文書名を抽出
                file_path = Path(doc.metadata.get('file_path', ''))
                doc_name = file_path.stem
                
                # メタデータに文書情報を追加
                doc.metadata.update({
                    'document_name': doc_name,
                    'source_file': file_path.name,
                    'content_type': 'Obsidian文書'
                })
                
                enhanced_documents.append(doc)
            
            print(f"📊 メタデータ強化完了: {len(enhanced_documents)}件")
            
            # ベクトルインデックスを構築（改良されたチャンク設定で）
            print("🔗 ベクトルインデックスを構築中...")
            self.index = VectorStoreIndex.from_documents(
                enhanced_documents,
                node_parser=self.node_parser,
                show_progress=False
            )
            
            # クエリエンジンを作成（RAG特化設定）
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=8,  # より多くの関連文書を検索
                response_mode="tree_summarize",  # 複数文書を効率的に要約
                verbose=False,  # 本番用はfalse
                node_postprocessors=[],  # 必要に応じて後処理を追加
            )
            
            print("✅ RAGエンジンの初期化が完了しました")
            print(f"📁 インデックス化されたノード数: {len(self.index.docstore.docs)}")
            return True
            
        except Exception as e:
            print(f"エラー: 日記ファイルの読み込みに失敗しました - {str(e)}")
            return False
    
    def answer_with_rag(self, query: str) -> str:
        """LlamaIndexによる完全なRAG処理（検索 + 生成）"""
        if not self.query_engine:
            return "エラー: RAGエンジンが初期化されていません"
        
        try:
            print(f"🔍 RAG処理開始: {query}")
            
            # LlamaIndexがドキュメント検索と回答生成を一括処理
            enhanced_query = f"""
            Obsidianの知識ベースから以下の質問に回答してください：

            質問: {query}

            回答の際は以下を心がけてください：
            - 関連する文書名があれば明記する
            - 具体的な内容や詳細を含める
            - 複数の文書にまたがる情報があれば統合して回答する
            - ソースとなる文書名を可能な限り示す
            - 会社の業務や経営に関する情報として適切にまとめる
            """
            
            response = self.query_engine.query(enhanced_query)
            
            print("✅ RAG処理完了")
            return str(response)
            
        except Exception as e:
            return f"RAG処理エラー: {str(e)}"
    
    def get_retrieval_info(self, query: str) -> dict:
        """検索でヒットしたドキュメント情報を取得（デバッグ用）"""
        if not self.query_engine:
            return {"error": "RAGエンジンが初期化されていません"}
        
        try:
            # Retrieverを直接使って検索情報を取得
            retriever = self.index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(query)
            
            result = {
                "query": query,
                "retrieved_count": len(nodes),
                "sources": []
            }
            
            for i, node in enumerate(nodes):
                source_info = {
                    "rank": i + 1,
                    "score": node.score,
                    "document_name": node.metadata.get('document_name', '不明'),
                    "source_file": node.metadata.get('source_file', '不明'),
                    "content_preview": node.text[:100] + "..." if len(node.text) > 100 else node.text
                }
                result["sources"].append(source_info)
            
            return result
            
        except Exception as e:
            return {"error": f"検索情報取得エラー: {str(e)}"}
    
    def is_ready(self) -> bool:
        """RAGエンジンが使用可能かチェック"""
        return self.query_engine is not None