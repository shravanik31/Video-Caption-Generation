import torch
import os
import json
import re
from torch.utils.data import DataLoader, Dataset
import numpy as np

def preprocess_data(filepath):
    with open(filepath, 'r') as f:
        file = json.load(f)

    word_counter = {}
    for data_entry in file:
        for caption in data_entry['caption']:
            words = re.sub('[.!,;?]', ' ', caption).split()
            for word in words:
                word = word.replace('.', '') if '.' in word else word
                if word in word_counter:
                    word_counter[word] += 1
                else:
                    word_counter[word] = 1

    filtered_words = {}
    for word, count in word_counter.items():
        if count > 3:
            filtered_words[word] = count

    useful_tokens = [('<PAD>', 0), ('<BOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    index_to_word = {i + len(useful_tokens): w for i, w in enumerate(filtered_words)}
    word_to_index = {w: i + len(useful_tokens) for i, w in enumerate(filtered_words)}

    for token, index in useful_tokens:
        index_to_word[index] = token
        word_to_index[token] = index
        
    print("Number of filtered words:", len(filtered_words))
    return index_to_word, word_to_index, filtered_words

def annotate_captions(label_file, filtered_words, word_to_index):
    label_json = label_file
    annotated_captions = []
    with open(label_json, 'r') as f:
        labels = json.load(f)
    for data_entry in labels:
        for caption in data_entry['caption']:
            sentence = re.sub(r'[.!,;?]', ' ', caption).split()
            for i, word in enumerate(sentence):
                if word not in filtered_words:
                    sentence[i] = 3
                else:
                    sentence[i] = word_to_index[word]
            sentence.insert(0, 1)
            sentence.append(2)
            annotated_captions.append((data_entry['id'], sentence))
    return annotated_captions

def create_minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

class DatasetWithFeatures(Dataset):
    def __init__(self, data_path, label_file, filtered_words, word_to_index):
        self.data_path = data_path
        self.filtered_words = filtered_words
        self.word_to_index = word_to_index
        self.avi_features = self.load_avi_features()
        self.data_pairs = annotate_captions(label_file, filtered_words, word_to_index)
        
    def load_avi_features(self):
        avi_features = {}
        features_path = self.data_path
        files = os.listdir(features_path)
        for file in files:
            feature_values = np.load(os.path.join(features_path, file))
            avi_features[file.split('.npy')[0]] = feature_values
        return avi_features

    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pairs[idx]
        feature_data = torch.Tensor(self.avi_features[avi_file_name])
        feature_data += torch.Tensor(feature_data.size()).random_(0, 2000)/10000.
        return torch.Tensor(feature_data), torch.Tensor(sentence)
    
    def print_insights(self, idx=None):
        unique_videos = set(pair[0] for pair in self.data_pairs)
        print(f"Number of unique videos: {len(unique_videos)}")

        avg_caption_length = np.mean([len(pair[1]) for pair in self.data_pairs])
        print(f"Average caption length: {avg_caption_length:.2f}")

        caption_lengths = [len(pair[1]) for pair in self.data_pairs]
        print(f"Caption length distribution: min={min(caption_lengths)}, max={max(caption_lengths)}, median={np.median(caption_lengths)}")
        
        index_to_word = {index: word for word, index in self.word_to_index.items()} 
        if idx is None:
            idx = np.random.randint(0, len(self.data_pairs)) 
        assert (idx < self.__len__()), f"Index {idx} is out of bounds."
        video_id, caption_indices = self.data_pairs[idx]
        caption_words = [index_to_word.get(index, '') for index in caption_indices]
        filtered_caption_words = [word for word in caption_words if word not in ('<BOS>', '<EOS>', '<UNK>', '<PAD>')]
        caption = ' '.join(filtered_caption_words).strip()
        print(f"Video ID: {video_id} and it's caption: {caption}")
    

class TestDataset(Dataset):
    def __init__(self, test_data_path):
        self.avi_features = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi_features.append([key, value])
    def __len__(self):
        return len(self.avi_features)
    def __getitem__(self, idx):
        return self.avi_features[idx]



