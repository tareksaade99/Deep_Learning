# Tweet Sentiment Classification and Relevant Word Identification

This project presents a comparative study between two approaches for **tweet sentiment classification** into *positive*, *neutral*, and *negative* categories:

- **TF-IDF + Logistic Regression** (baseline model)  
- **Fine-tuned BERT (bert-base-uncased)** (modern transformer model)

The dataset comes from **Figure Eight’s Data for Everyone** platform and includes labeled tweets reflecting user sentiment.

---

## 🔍 Methodology

### Data Preprocessing
Tweets were cleaned by removing URLs, mentions, and non-alphanumeric characters, while keeping emoticons such as `:)` or `:-(` due to their strong link to sentiment.  
The data was split into **70% training**, **15% validation**, and **15% test** sets.

### Models
- **TF-IDF + Logistic Regression:** uses unigrams and bigrams to convert text into weighted numerical features.  
- **BERT Fine-tuning:** adapts the pre-trained BERT model for sentiment classification through three training epochs (learning rate `2e-5`, batch size `64`).

---

## 📊 Results

| Model | Validation F1 | Test F1 | Inference Time |
|--------|---------------|---------|----------------|
| TF-IDF | 0.69 | 0.68 | 0.14s |
| BERT | **0.78** | **0.78** | 11.87s |

BERT significantly outperforms the baseline in accuracy but requires **~85× more inference time**.

---

## 🧩 Bonus Task: Sentiment-Relevant Word Identification

Two interpretability methods were explored:
1. **Attention-based token selection**
2. **Integrated Gradients attribution**

Performance was evaluated using **Jaccard similarity** between selected and human-annotated words.  
Both methods achieved low maximum similarity (≈0.16), suggesting challenges due to annotation inconsistencies and the subjective nature of sentiment.

---

## 💡 Discussion & Future Work

- BERT offers strong performance but high computational cost.  
- The relevant-word task remains difficult; annotation quality and evaluation metrics limit performance assessment.  
- Future directions:
  - Use **Twitter-specific models** (e.g., BERTweet)  
  - Explore **F1-overlap** or **ROUGE** metrics  
  - Investigate **sequence labeling** or **span prediction** architectures  

---

## 🧠 Key Takeaways

- BERT outperforms TF-IDF for sentiment classification on Twitter data.  
- Identifying sentiment-relevant words is more complex than classification itself.  
- Better evaluation and modeling strategies are needed for interpretability tasks.
