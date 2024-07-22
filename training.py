import torch
from torch.autograd import Variable
import torch.optim as optim
import pickle
from data_preprocessing import DatasetWithFeatures, create_minibatch, preprocess_data 
from models import Seq2SeqModel, EncoderRNN, DecoderRNN
from torch.utils.data import DataLoader
import torch.nn as nn
import time 
import matplotlib.pyplot as plt 
import sys


def train_model(model, epoch, loss_fn, parameters, optimizer, train_loader):
    model.train()
    start_time = time.time()
    epoch_loss = 0  
    
    for batch_idx, batch in enumerate(train_loader):
        avi_features, ground_truths, lengths = batch
        avi_features, ground_truths = avi_features.cuda(), ground_truths.cuda()
        avi_features, ground_truths = Variable(avi_features), Variable(ground_truths)
        
        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_features, target_sentences=ground_truths, mode='train', tr_steps=epoch)
        ground_truths = ground_truths[:, 1:]

        batch_size = len(seq_logProb)
        concatenated_predictions = torch.cat([seq_logProb[i][:lengths[i]-1] for i in range(batch_size)], dim=0)
        concatenated_ground_truths = torch.cat([ground_truths[i][:lengths[i]-1] for i in range(batch_size)], dim=0)
        loss = loss_fn(concatenated_predictions, concatenated_ground_truths) / batch_size

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  

    end_time = time.time()
    epoch_time = end_time - start_time
    avg_epoch_loss = epoch_loss / len(train_loader) 

    print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss))
    return epoch_loss, epoch_time

def test_model(test_loader, model, index_to_word, beam_size = 5):
    model.eval()
    test_results = []
    for batch_idx, batch in enumerate(test_loader):
        video_ids, avi_features = batch
        avi_features = avi_features.cuda()
        video_ids, avi_features = video_ids, Variable(avi_features).float()

        seq_logProb, seq_predictions = model(avi_features, mode='inference')
        test_predictions = seq_predictions
        
        result = [[index_to_word[x.item()] if index_to_word[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        video_results = zip(video_ids, result)
        for video_result in video_results:
            test_results.append(video_result)
    return test_results


def main():
    print("To initiate the training process, please specify the path to your training data features as the first argument and the path to your 'training_label.json' file as the second argument.")
    files_dir = sys.argv[1]
    label_file = sys.argv[2]  
    index_to_word, word_to_index, filtered_words = preprocess_data(sys.argv[2])
    with open('index_to_word.pickle', 'wb') as handle:
        pickle.dump(index_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    train_dataset = DatasetWithFeatures(files_dir, label_file, filtered_words, word_to_index)  
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8, collate_fn=create_minibatch)
    train_dataset.print_insights(idx=10)

    
    epochs_n = 200

    encoder = EncoderRNN()
    decoder = DecoderRNN(512, len(index_to_word) + 4, len(index_to_word) + 4, 1024, 0.3)
    model = Seq2SeqModel(encoder_instance=encoder, decoder_instance=decoder)
    
    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.001)

    losses = [] 
    total_time = 0  
    
    for epoch in range(epochs_n):
        epoch_loss, epoch_time = train_model(model, epoch + 1, loss_fn, parameters, optimizer, train_dataloader)
        losses.append(epoch_loss)
        total_time += epoch_time  

    torch.save(model, "{}.h5".format('model_shravani'))
    print("Training finished")
    print("Total training time: {:.2f} seconds".format(total_time))
    
    with open('losses_array.txt', 'w') as f:
            for loss in losses:
                f.write(f"{loss}\n")

if __name__ == "__main__":
    main()
