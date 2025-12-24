#!/usr/bin/env python3
"""
株式会社AYMEN Obsidian RAG搭載 AIチャットボット
LangChain + LlamaIndexによるハイブリッドRAGシステム
"""

import sys
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from engine.rag_engine import RAGEngine
from config.settings import Settings

class CompanyBot:
    """LangChain（判定・ルーティング） + LlamaIndex（RAG）ハイブリッドボット - 会社情報専用"""
    
    def __init__(self):
        """チャットボットを初期化"""
        self.rag_engine = RAGEngine()
        self.classifier_llm = None
        self.general_llm = None
        self._setup_chains()
    
    def _setup_chains(self) -> bool:
        """LangChainの設定（判定と一般対話用）"""
        try:
            # 判定用LLM（一貫性重視）
            self.classifier_llm = ChatOpenAI(
                api_key=Settings.OPENROUTER_API_KEY,
                base_url=Settings.OPENROUTER_BASE_URL,
                model=Settings.MODEL_NAME,
                temperature=0.1
            )
            
            # 一般対話用LLM（自然な会話）
            self.general_llm = ChatOpenAI(
                api_key=Settings.OPENROUTER_API_KEY,
                base_url=Settings.OPENROUTER_BASE_URL,
                model=Settings.MODEL_NAME,
                temperature=0.7
            )
            
            return True
            
        except Exception as e:
            print(f"エラー: LLMの初期化に失敗しました - {str(e)}")
            return False
    
    def _classify_question(self, question: str) -> str:
        """質問を分類（RAG vs GENERAL）"""
        try:
            classifier_prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは質問を分析して、適切な処理方法を判定するAIです。

以下の基準で判定してください：

【RAG】- 以下の場合は「RAG」と回答：
- 会社の事業内容について質問している
- 役員報酬や給与について聞いている
- クレジットカードや経費について質問している
- 理念や経営方針について聞いている
- 会社の業務や制度に関する質問
- 株式会社AYMENに関連する質問

【GENERAL】- 以下の場合は「GENERAL」と回答：
- 挨拶やあいさつ
- 一般的な知識や情報を求めている
- 雑談や世間話
- 現在や未来に関する質問
- 会社とは無関係な質問

「RAG」または「GENERAL」のどちらかのみを回答してください。"""),
                ("human", "{question}")
            ])
            
            chain = classifier_prompt | self.classifier_llm
            result = chain.invoke({"question": question})
            classification = result.content.strip().upper()
            print(f"🤔 質問分類: {question} → {classification}")
            return classification
        except Exception as e:
            print(f"分類エラー: {str(e)}")
            return "GENERAL"  # デフォルトは一般対話
    
    def _answer_general(self, question: str) -> str:
        """LangChainによる一般対話"""
        try:
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", """あなたは親しみやすいAIアシスタントです。
ユーザーと自然で楽しい対話を心がけ、日本語で親しみやすく回答してください。
挨拶、雑談、一般的な知識に関する質問に答えてください。"""),
                ("human", "{question}")
            ])
            
            chain = general_prompt | self.general_llm
            result = chain.invoke({"question": question})
            return result.content
        except Exception as e:
            return f"一般対話エラー: {str(e)}"
    
    def initialize(self) -> bool:
        """チャットボットの初期化"""
        print("🤖 株式会社AYMEN情報チャットボットを起動中...")
        
        # 設定の検証
        if not Settings.validate_settings():
            return False
        
        Settings.print_settings()
        
        # RAGエンジンの初期化
        if not self.rag_engine.load_documents():
            print("警告: RAGエンジンの初期化に失敗しました。一般的な対話のみ利用可能です。")
        
        if not self.classifier_llm or not self.general_llm:
            print("エラー: LLMの初期化に失敗しました")
            return False
        
        return True
    
    def chat(self):
        """メインチャットループ"""
        print("\n✨ チャットボットが準備できました！")
        print("💡 'exit' または 'quit' で終了します")
        print("🔍 'debug:質問' で検索詳細を表示します")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nUser: ").strip()
                
                if user_input.lower() in ['exit', 'quit', '終了']:
                    print("\n👋 チャットボットを終了します。")
                    break
                
                if not user_input:
                    continue
                
                # 1. LangChainで質問を分類
                classification = self._classify_question(user_input)
                
                # 2. 分類結果に基づいてルーティング
                if classification == "RAG" and self.rag_engine.is_ready():
                    # LlamaIndexでRAG処理
                    response = self.rag_engine.answer_with_rag(user_input)
                    print(f"\nBot (RAG): {response}")
                    
                    # デバッグ: どの文書が参照されたかを表示
                    if user_input.startswith("debug:"):
                        debug_query = user_input[6:].strip()
                        retrieval_info = self.rag_engine.get_retrieval_info(debug_query)
                        print(f"\n🔍 検索情報:")
                        for source in retrieval_info.get("sources", []):
                            print(f"  📄 {source['document_name']} (スコア: {source['score']:.3f})")
                            print(f"     {source['content_preview']}")
                else:
                    # LangChainで一般対話
                    response = self._answer_general(user_input)
                    print(f"\nBot (一般): {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 チャットボットを終了します。")
                break
            except Exception as e:
                print(f"\nエラーが発生しました: {str(e)}")
                print("続行します...")

def main():
    """メイン関数"""
    bot = CompanyBot()
    
    if not bot.initialize():
        print("チャットボットの初期化に失敗しました。")
        sys.exit(1)
    
    bot.chat()

if __name__ == "__main__":
    main()