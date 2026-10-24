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

## Fathima's Contribution — Model Development and Evaluation

I developed and evaluated Malayalam-to-English sequence-to-sequence translation models using TensorFlow/Keras and the preprocessed dataset.

**Model development:**

- Built and trained a baseline LSTM encoder–decoder model.
- Built and trained an attention-based encoder–decoder model using additive (Bahdanau-style) attention.
- Used the Malayalam and English tokenizers from the preprocessing stage, adding start and end tokens to English decoder sequences.
- Saved the trained attention model weights as `attention.weights.h5`.

**Evaluation:**

- Evaluated the models on the validation set using loss and token accuracy, including masked accuracy that excludes padding tokens.
- Compared baseline and attention model training histories using loss and accuracy plots.
- Tested sample translations using greedy, word-by-word decoding.

  ## Fine-Tuned OPUS-MT Malayalam-to-English Model

In addition to the LSTM models, this project fine-tunes the pretrained
[Helsinki-NLP/opus-mt-ml-en](https://huggingface.co/Helsinki-NLP/opus-mt-ml-en)
model for Malayalam-to-English translation. Fine-tuning adapts the pretrained
translation model using the project’s Malayalam-English sentence pairs.

### Training

The model was fine-tuned using `data/train.csv` and evaluated during training
using `data/validation.csv`. Both files contain `malayalam` and `english`
columns. Training ran for 3 epochs with a batch size of 8, a learning rate of
`5e-5`, and maximum source and target sequence lengths of 128 tokens.

The final reported validation metrics were:

| Metric | Result |
|---|---:|
| Validation BLEU | 27.57 |
| Validation loss | 1.866 |

BLEU measures how closely generated translations match reference translations;
it is **not an accuracy percentage**. These results are from the validation
set, not the held-out test set.

### Running the interactive translator

After installing the required PyTorch and Transformers dependencies and placing
the saved model in `opus_mt_finetuned/`, run:

```bash
python translate_opus_mt.py
