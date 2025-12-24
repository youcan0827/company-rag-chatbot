import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """アプリケーション設定を管理するクラス"""
    
    # APIキー
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # モデル設定
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    # データパス
    DOCS_PATH = os.getenv("DOCS_PATH", "/Users/yoshinomukanou/Downloads/国交省に関するデータ")
    
    # OpenRouterのベースURL
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    @classmethod
    def validate_settings(cls) -> bool:
        """設定の妥当性を検証"""
        if not cls.OPENROUTER_API_KEY:
            print("エラー: OPENROUTER_API_KEYが設定されていません")
            return False
        
        docs_path = Path(cls.DOCS_PATH)
        if not docs_path.exists() or not docs_path.is_dir():
            print(f"エラー: データパスが見つかりません: {cls.DOCS_PATH}")
            return False
        
        return True
    
    @classmethod
    def print_settings(cls):
        """設定情報を表示（APIキーは隠す）"""
        print("=== 設定情報 ===")
        print(f"モデル: {cls.MODEL_NAME}")
        print(f"データパス: {cls.DOCS_PATH}")
        print(f"APIキー: {'設定済み' if cls.OPENROUTER_API_KEY else '未設定'}")
        print("===============")