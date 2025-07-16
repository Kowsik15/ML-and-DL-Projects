# Human Behavior Classification using Deep Learning 🧠

This project focuses on classifying human behavior into categories using Deep Learning techniques. The model is trained on labeled text data (tweets, reviews, or similar) and uses an RNN/LSTM-based architecture for sequence modeling.

---

## 📌 Project Objective
To build a multi-class classification model that can identify the **emotional or behavioral tone** of a given input text as:
- Positive
- Negative
- Neutral

---

## 🛠️ Tools & Technologies
- Python
- TensorFlow / Keras
- NumPy, Pandas
- NLTK / SpaCy (for preprocessing)
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📂 Dataset Description
- Format: CSV
- Columns:
  - `Text`: The input sentence or tweet
  - `Label`: Class label (Positive/Negative/Neutral)

---

## 🔄 Workflow

1. **Data Preprocessing**  
   - Lowercasing, punctuation removal, tokenization  
   - Padding and sequence encoding with `Tokenizer`  

2. **Model Architecture**  
   - Embedding layer  
   - LSTM layer  
   - Dense (softmax) output  

3. **Training & Evaluation**  
   - Loss Function: `categorical_crossentropy`  
   - Optimizer: `Adam`  
   - Metrics: Accuracy  

---

## 🧪 Results

| Metric       | Score     |
|--------------|-----------|
| Train Accuracy | 90%+ |
| Validation Accuracy | ~88% |
| Confusion Matrix | ✔️ (included in notebook) |

---

## 📈 Visuals

> Add 1-2 screenshots of training accuracy/loss graph, confusion matrix

---

## 🧠 Key Learnings
- Applied deep learning to NLP classification
- Hands-on with sequence preprocessing using `Tokenizer` & `pad_sequences`
- Gained experience in hyperparameter tuning and dropout regularization

---

## 📁 Repository Structure