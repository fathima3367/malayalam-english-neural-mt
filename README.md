# Malayalam–English Neural Machine Translation

## Project Overview
This project aims to translate Malayalam sentences into English using a Neural Machine Translation (NMT) system based on an Encoder–Decoder architecture with LSTM/GRU.

## Team Members
- Fariza Noushad
- Fathima M J
- Shifana E M

## Fariza's Contribution — Dataset Preprocessing
I prepared the Malayalam–English parallel dataset using Python and Google Colab.

**Dataset:** English–Malayalam Bi-Parallel Corpus

**Preprocessing steps:**
- Loaded and aligned Malayalam and English sentence files.
- Removed whitespace, empty sentence pairs, and exact duplicates.
- Filtered sentences containing 2–50 words.
- Split the dataset into training, validation, and testing sets.
- Tokenized both languages using separate Keras tokenizers.
- Converted sentences into numerical sequences and applied padding.

## Dataset Statistics

| Dataset | Sentence pairs |
|---|---:|
| Training | 116,016 |
| Validation | 14,502 |
| Testing | 14,503 |
| **Total** | **145,021** |

## Technologies Used
Python, Google Colab, Pandas, NumPy, Scikit-learn, TensorFlow/Keras.

## Preprocessing Outputs
- `cleaned_dataset.csv`
- `train.csv`
- `validation.csv`
- `test.csv`
- `ml_tokenizer.pkl`
- `en_tokenizer.pkl`
- `preprocessed_data.npz`

## Project Status
Dataset preprocessing is completed. Encoder–Decoder model development and translation evaluation will be documented as the project progresses.
