# Video Caption Generation

## Overview

The goal for this project is to develop a model that can automatically generate descriptive captions for short videos, considering the diverse attributes of the videos, such as varying objects and actions, and handling the variable length of both the input videos and the output captions.

## Dataset

The dataset used for this project is the Microsoft Video Description Corpus (MSVD), which consists of 1970 short video clips from YouTube, each accompanied by multiple human-generated English captions. For our assignment, we utilize a subset of this dataset, comprising 1450 videos for training and 100 videos for testing. The videos cover a wide range of content, from sports and animals to everyday activities, providing a diverse set of visual and contextual scenarios for caption generation.

## Data Preprocessing

1. **`preprocess_data` function**:
    - Reads a JSON file containing video captions.
    - Builds a dictionary of words based on their frequency.
    - Filters out words with a count less than three.
    - Creates mappings between words and indices, including special tokens for padding, beginning of sentence, end of sentence, and unknown words.

2. **Special Tokens**:
    - `<PAD>`: Used to pad the sentence to the equal length.
    - `<BOS>`: Beginning of sentence, a sign to start generating the output sentence.
    - `<EOS>`: End of sentence, a sign at the end of the output sentence.
    - `<UNK>`: Used when the word is not present in the dictionary or to ignore the unknown word.

3. **`annotate_captions` function**:
    - Converts captions into sequences of indices based on the filtered dictionary.

4. **`create_minibatch` function**:
    - Sorts and pads the captions to create minibatches for training.

5. **`DatasetWithFeatures` class**:
    - Loads video features and pairs them with the processed captions for training.

6. **`TestDataset` class**:
    - Loads features for testing videos.

## Model Building

1. **EncoderRNN**:
    - Processes the input video features.
    - Applies a linear projection to reduce the feature dimension to the hidden size of the Gated Recurrent Units (GRU) layer.
    - Applies dropout for regularization.
    - Passes the processed features through a GRU layer to produce a sequence of hidden states.

2. **Attention**:
    - Calculates the context vector for each time step in the decoder.
    - Takes the decoder's current hidden state and the encoder's outputs as inputs.
    - Applies a softmax function to obtain attention weights.
    - Computes a weighted sum of the encoder outputs, resulting in the context vector.

3. **DecoderRNN**:
    - Generates the output caption.
    - Uses an embedding layer to convert input word indices into dense vectors.
    - Combines dense vectors with the context vector from the attention module.
    - Uses a GRU layer to update the decoder's hidden state.
    - Predicts the next word in the caption using a linear layer.
    - Supports teacher forcing during training.

4. **Seq2SeqModel**:
    - Takes video features as input.
    - Passes them through the encoder to get the encoder outputs and last hidden state.
    - Feeds these into the decoder to generate the output caption.
    - Operates in two modes: 'train' for training and 'inference' for generating captions without ground truth inputs.

## Training

- **Command**:
    ```bash
    python3 training.py /path/to/training_data/feat /path/to/training_label.json
    ```
    - Ensure you specify your respective paths to the training data features folder and training_label.json file.

- **Parameters**:
    - Number of epochs: 200
    - Batch size: 128
    - Learning rate: 0.001
    - Optimizer: Adam
    - Loss function: Cross-Entropy Loss

- **Output**:
    - A trained model saved as `model_shravani.h5`.

## Testing

- **Preparation**:
    - Before testing, download the pretrained model model_shravani.h5, testing_label.json, index_to_word.pickle, and blue_eval.py files to your respective directory. To test the model, run the following command:

- **Command**:
    ```bash
    sh hw2_seq2seq.sh /path/to/testing_data output_captions.txt
    ```
    - Specify your respective paths to the testing data folder and output_captions.txt file.

- **Output**:
    - Generated captions stored in an output file.
    - BLEU score calculated by comparing the generated captions with the ground truth captions.

## Results

- The average BLEU score obtained is 0.625.
- [Download Model](https://drive.google.com/file/d/19dvvQgTKG4UelaMULE6lmvMWQ06yPlLJ/view?usp=sharing)

## Dependencies

- Python 3.x
- CUDA
- PyTorch
- NumPy
- SciPy
- NLTK
- scikit-learn
- json
