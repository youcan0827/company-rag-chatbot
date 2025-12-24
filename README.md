# 🤖 Obsidian RAG搭載 AIチャットボット

LangChain + LlamaIndexを組み合わせたハイブリッドRAGシステムにより、個人のObsidian日記データを参照可能なAIチャットボットです。

## 🎯 機能

- **インテリジェント・ルーティング**: 一般的な対話と日記検索を自動判断
- **日記検索**: Obsidianの日記ファイルから関連情報を検索・要約
- **自然な対話**: 親しみやすいAIアシスタントとしての応答
- **メモリ機能**: 直近の会話履歴を記憶

## 📁 プロジェクト構造

```
.
├── main.py              # エントリポイント（LangChain Agent）
├── engine/
│   └── rag_engine.py    # LlamaIndex検索ロジック
├── config/
│   └── settings.py      # 設定管理
├── .env                 # APIキー設定
├── requirements.txt     # 依存ライブラリ
└── README.md           # このファイル
```

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルを編集してOpenRouterのAPIキーを設定：

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
DIARY_PATH=/Users/yoshinomukanou/Documents/Obsidian Vault/日記
MODEL_NAME=openai/gpt-4o-mini
```

### 3. 実行

```bash
python main.py
```

## 💡 使用方法

### 基本的な対話
```
User: こんにちは！
Bot: こんにちは！何かお手伝いできることはありますか？
```

### 日記検索
```
User: 先月はどんなことがあった？
Bot: （日記から関連情報を検索して回答）

User: 2025年12月22日の日記を教えて
Bot: （該当日の日記内容を要約して回答）
```

### 終了
```
User: exit
```

## 🔧 技術スタック

- **Python 3.10+**
- **LangChain**: 質問分類とルーティング、一般対話
- **LlamaIndex**: 完全なRAG処理（検索 + 生成）
- **OpenRouter**: GPT-4o-mini API
- **OpenAI Embeddings**: テキストベクトル化

## 📝 アーキテクチャ詳細

### ハイブリッドシステム構成
```
User質問 → LangChain分類 → ルーティング
           ↓                ↓
    [一般質問]          [RAG質問]
       ↓                 ↓
   LangChain対話     LlamaIndex RAG
   (temperature=0.7)  (検索+生成)
```

### データソース
- **対象**: `/Users/yoshinomukanou/Documents/Obsidian Vault/日記`内の`.md`ファイル
- **処理**: 起動時に全ファイルをスキャンしてベクトルインデックス構築

### 質問分類ロジック
- **RAG処理**: 過去の出来事、日付指定、感情・思考、日記内容
- **一般対話**: 挨拶、雑談、一般知識、現在・未来の質問

### LLM設定の最適化
- **分類用**: temperature=0.1（一貫性重視）
- **一般対話**: temperature=0.7（自然な会話）
- **RAG**: temperature=0.2（事実ベース、system_prompt付き）

## ⚠️ 注意事項

1. **APIキー**: OpenRouterのAPIキーが必要
2. **データパス**: Obsidianの日記パスが正しく設定されている必要あり
3. **ネットワーク**: インターネット接続必須（OpenRouter API使用のため）

## 🔍 トラブルシューティング

### APIキーエラー
```
エラー: OPENROUTER_API_KEYが設定されていません
```
→ `.env`ファイルでAPIキーを設定してください

### 日記パスエラー
```
エラー: 日記パスが見つかりません
```
→ `.env`ファイルでDIARY_PATHを正しいパスに設定してください