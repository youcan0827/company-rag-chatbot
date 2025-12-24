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
    DIARY_PATH = os.getenv("DIARY_PATH", "/Users/yoshinomukanou/Documents/Obsidian Vault/株式会社AYMEN")
    
    # OpenRouterのベースURL
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    
    @classmethod
    def validate_settings(cls) -> bool:
        """設定の妥当性を検証"""
        if not cls.OPENROUTER_API_KEY:
            print("エラー: OPENROUTER_API_KEYが設定されていません")
            return False
        
        diary_path = Path(cls.DIARY_PATH)
        if not diary_path.exists() or not diary_path.is_dir():
            print(f"エラー: データパスが見つかりません: {cls.DIARY_PATH}")
            return False
        
        return True
    
    @classmethod
    def print_settings(cls):
        """設定情報を表示（APIキーは隠す）"""
        print("=== 設定情報 ===")
        print(f"モデル: {cls.MODEL_NAME}")
        print(f"データパス: {cls.DIARY_PATH}")
        print(f"APIキー: {'設定済み' if cls.OPENROUTER_API_KEY else '未設定'}")
        print("===============")