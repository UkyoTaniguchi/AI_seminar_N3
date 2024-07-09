from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# 1. データセットの読み込み
dataset = load_dataset('wrime')

# 2. トークナイザーとモデルの読み込み
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', num_labels=3)

# 3. データの前処理
def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. 学習設定
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 5. モデルの学習
trainer.train()

# 6. 感情分析関数の定義
def analyze_sentiment(review):
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=-1)
    sentiment = probabilities.argmax().item()
    return sentiment

# サンプルレビューの分析
sample_review = "この商品は本当に素晴らしい！とても満足しています。"
sentiment = analyze_sentiment(sample_review)
print(f"Sentiment: {sentiment}")


# サンプルレビューのリスト
sample_reviews = [
    "この商品は本当に素晴らしい！とても満足しています。",
    "品質が悪く、期待外れでした。お勧めしません。",
    "まあまあの製品ですが、価格に見合った価値があると思います。",
    "配送が遅れましたが、製品自体は良かったです。",
    "カスタマーサービスがとても親切で、問題をすぐに解決してくれました。"
]

# 感情分析を行う関数
def analyze_sentiments(reviews):
    results = []
    for review in reviews:
        sentiment = analyze_sentiment(review)
        results.append((review, sentiment))
    return results

# サンプルレビューの感情分析
analyzed_reviews = analyze_sentiments(sample_reviews)

# 結果の表示
for review, sentiment in analyzed_reviews:
    sentiment_label = ["ネガティブ", "ニュートラル", "ポジティブ"][sentiment]
    print(f"Review: {review}\nSentiment: {sentiment_label}\n")
