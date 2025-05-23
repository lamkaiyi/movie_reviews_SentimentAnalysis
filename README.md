# movie_reviews_SentimentAnalysis

# 🎬 Sentiment Analysis on Rotten Tomatoes Movie Reviews

This project explores sentiment classification using the **Rotten Tomatoes Movie Review** dataset. The task is to predict the sentiment (positive/negative) of short movie reviews. Three different modeling approaches were evaluated to compare their effectiveness in sentiment analysis.

## 📂 Dataset

We used the **Rotten Tomatoes** dataset from the [GLUE benchmark](https://gluebenchmark.com/tasks) which consists of short movie reviews with sentiment labels:

- **Positive**
- **Negative**

The dataset is preprocessed into training, validation, and test splits.

## 🧠 Models and Methods

Three different deep learning approaches were implemented and compared:

| Model                          | Embedding Method | Accuracy on Test Dataset |
|-------------------------------|------------------|----------|
| LSTM                          | [FastText](https://fasttext.cc/docs/en/english-vectors.html) | 74%      |
| LSTM                          | BERT             | TBC      |
| BERT (Transformer-based Model)| BERT             | TBC      |

### 🔧 Details

- **LSTM + FastText**: Traditional recurrent architecture using pre-trained [FastText](https://fasttext.cc/docs/en/english-vectors.html) embeddings for word representation.
- **LSTM + BERT**: Leveraged BERT-based embeddings as input features to an LSTM model for enhanced contextual understanding.
- **BERT Model**: Fine-tuned the pre-trained BERT base model directly on the sentiment classification task.

