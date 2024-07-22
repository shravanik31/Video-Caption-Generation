import sys
import torch
import json
from models import EncoderRNN, DecoderRNN, Attention, Seq2SeqModel
from training import test_model
from data_preprocessing import TestDataset
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle

def load_model(model_path):
    return torch.load(model_path, map_location=lambda storage, loc: storage)

def load_index_to_word(i2w_path):
    with open(i2w_path, 'rb') as handle:
        return pickle.load(handle)

def write_predictions(test_loader, model, index_to_word, output_file):
    predictions = test_model(test_loader, model, index_to_word)
    with open(output_file, 'w') as f:
        for video_id, caption in predictions:
            f.write('{},{}\n'.format(video_id, caption))

def calculate_bleu_score(test_label_path, output_file):
    test_data = json.load(open(test_label_path))
    result = {}
    with open(output_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            comma_index = line.index(',')
            test_id = line[:comma_index]
            caption = line[comma_index+1:]
            result[test_id] = caption

    bleu_scores = []
    for item in test_data:
        captions = [caption.rstrip('.') for caption in item['caption']]
        bleu_scores.append(BLEU(result[item['id']], captions, True))

    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return average_bleu_score

if __name__ == "__main__":
    print("Please download the model, index-to-word pickle, and 'testing_label.json' to run the shell script.")
    model_path = 'model_shravani.h5'
    model = load_model(model_path)
    model = model.cuda()

    test_features_path = sys.argv[1]+'/feat'
    test_dataset = TestDataset(test_features_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=8)

    index_to_word_path = 'index_to_word.pickle'
    index_to_word = load_index_to_word(index_to_word_path)

    output_file = sys.argv[2]
    write_predictions(test_loader, model, index_to_word, output_file)

    test_label_path = 'testing_label.json'
    average_bleu_score = calculate_bleu_score(test_label_path, output_file)
    print("Average BLEU score is", average_bleu_score)