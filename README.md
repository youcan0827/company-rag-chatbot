# 🏢 国土交通省データ RAG搭載 AIチャットボット

LangChain + LlamaIndexを組み合わせたハイブリッドRAGシステムにより、国土交通省のPDF文書データを参照可能なAIチャットボットです。

## 🎯 機能

- **インテリジェント・ルーティング**: 一般的な対話と国交省文書検索を自動判断
- **国交省データ検索**: 国土交通省のPDF文書から関連情報を検索・要約
- **政策情報取得**: インフラ、交通、国土計画などの情報を提供
- **自然な対話**: 親しみやすいAIアシスタントとしての応答

## 📁 プロジェクト構造

```
.
├── main.py              # エントリポイント（MlitBot）
├── rag_engine.py        # LlamaIndex検索ロジック
├── settings.py          # 設定管理
├── .env                 # APIキー設定（gitignore済み）
├── .gitignore           # Git除外ファイル
├── requirements.txt     # 依存ライブラリ
└── README.md           # このファイル
```

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env`ファイルを作成してOpenRouterのAPIキーを設定：

```env
# OpenRouterのAPIキーを設定してください
OPENROUTER_API_KEY=your_openrouter_api_key_here

# 国交省データファイルのパス
DOCS_PATH=/Users/yoshinomukanou/Downloads/国交省に関するデータ

# 使用モデル（OpenRouter経由のGPT-4o-mini）
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

### 国交省データ検索
```
User: 国土交通省のインフラ政策について教えて
Bot: （国交省PDF文書から関連情報を検索して回答）

User: 交通政策の最新動向は？
Bot: （該当する政策文書を要約して回答）

# デバッグモード（検索元文書を表示）
User: debug:国交省のインフラ政策について教えて
Bot: （検索結果と参照した文書名を表示）
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
- **対象**: `/Users/yoshinomukanou/Downloads/国交省に関するデータ`内のPDFファイル
- **ファイル形式**: PDFファイルのみを読み込み
- **処理**: 起動時に全PDFをスキャンしてベクトルインデックス構築
- **チャンクサイズ**: 1024トークン（PDF文書に最適化）

### 質問分類ロジック
- **RAG処理**: 国交省政策、インフラ、交通、法令・制度、国土計画
- **一般対話**: 挨拶、雑談、一般知識、国交省と無関係な質問

### LLM設定の最適化
- **分類用**: temperature=0.1（一貫性重視）
- **一般対話**: temperature=0.7（自然な会話）
- **RAG**: temperature=0.2（事実ベース、system_prompt付き）

## ⚠️ 注意事項

1. **APIキー**: OpenRouterのAPIキーが必要（https://openrouter.ai で取得）
2. **データファイル**: 国交省PDFデータが指定パスに必要
3. **Pythonバージョン**: Python 3.10+推奨
4. **ネットワーク**: インターネット接続必須（OpenRouter API使用のため）

## 🔍 トラブルシューティング

### APIキーエラー
```
エラー: OPENROUTER_API_KEYが設定されていません
```
→ `.env`ファイルでAPIキーを設定してください

### データパスエラー
```
エラー: データパスが見つかりません
```
→ `.env`ファイルでDOCS_PATHを正しいパスに設定してください

### PDF読み込みエラー
```
警告: PDFファイルが見つかりません
```
→ 指定ディレクトリに国交省のPDFファイルを配置してください

### 初期化時間が長い
初回のPDF読み込み時は数分かかる場合があります。しばらくお待ちください。