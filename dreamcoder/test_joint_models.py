"""
test_joint_models: Author : Catherine Wong

Tests joint model implementations. For now, a scratch interface for development.
"""
# Support for https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------------------
import sys

# ------------------- FLAGS -------------------
USE_ATTENTION = True  # TODO: verify 
USE_BERT = False      # TODO: implement
OUTPUT_PROGs = False  # else output language

bert_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased') # https://pytorch.org/hub/huggingface_pytorch-transformers/
# ------------------- FLAGS -------------------

# Helper function; parses a list of tasks into a list of pure (input, output) sentence pairs
def parse_tasks_into_pairs(result, tasks):
    pairs = []
    for task in tasks:
        task_name = task.name
        task_language = result.taskLanguage[task_name]
        groundTruthProgram = task.groundTruthProgram 
        ground_truth_program_tokens = task.groundTruthProgram.left_order_tokens(show_vars=True) # A good example of how to turn programs into sequences as a baseline. Removes variables -- you could put this back. See program.py - line: 77
        # print(f"Task name: {task_name}")
        # print(f"Task language:  {task_language}")
        # print(f"Ground truth program: {groundTruthProgram}")
        # print(f"Ground truth program tokens: {ground_truth_program_tokens}")
        # print()
        # Additional attributes on LOGO tasks: see makeLogoTasks.py
        # task.highresolution <== an array containing the image.
        for task_lang in task_language:
            # pairs must be of the form [(input sentence, output sentence)]
            if OUTPUT_PROGs:
                pairs.append([task_lang, ' '.join(ground_truth_program_tokens)])
            else:
                pairs.append([' '.join(ground_truth_program_tokens), task_lang])

    return pairs 

# Helper function; parses a list of in/out sentence pairs into two language objects
def prepare_data(in_out_pairs):
        prog_lang, nl_lang = None, None 

        if USE_BERT:
            print("using BERT tokenizer")
            prog_lang, nl_lang = BERTLang("prog", bert_tokenizer), BERTLang("lang", bert_tokenizer)
        else:
            print("using baseline tokenizer")
            prog_lang, nl_lang = Lang("prog"), Lang("lang")
        
        for pair in in_out_pairs:
            if OUTPUT_PROGs:
                nl_lang.addSentence(pair[0])
                prog_lang.addSentence(pair[1])
            else:
                prog_lang.addSentence(pair[0])
                nl_lang.addSentence(pair[1])
        
        print(f"Data Prepared | Found\n{prog_lang.name, prog_lang.n_words}\n{nl_lang.name, nl_lang.n_words}")
        if OUTPUT_PROGs:
            return nl_lang, prog_lang
        else:
            return prog_lang, nl_lang


# !!! MAIN ENTRY !!!
# On command line: python3 bin/logo.py --enumerationTimeout 1 --testingTimeout 1 --iterations 12 --biasOptimal --contextual --taskBatchSize 40 --testEvery 1 --no-cuda --recognitionTimeout 1800 --recognition_0 --recognition_1 examples language --Helmholtz 0.5 --synchronous_grammar --skip_first_test --taskDataset logo_unlimited_200 --language_encoder recurrent --languageDataset logo_unlimited_200/human --sample_n_supervised 0 --moses_dir ./moses_compiled --smt_phrase_length 1 --language_compression --lc_score 0.2 --max_compression 5 --smt_pseudoalignments 0.0 --max_mem_per_enumeration_thread 5000000000 --seed 1 --test_joint_language_program_model       
def test_joint_language_program_model(result, train_tasks, testing_tasks):
    # All tasks have a ground truth program and a name
    print(f'Length train tasks: {len(train_tasks)}')
    print(f'Length test tasks: {len(testing_tasks)}')
    pairs = parse_tasks_into_pairs(result, train_tasks)
    
    # Data Prep
    input_lang, output_lang = prepare_data(pairs)
    MAX_LENGTH = max(input_lang.max_sentence_length, output_lang.max_sentence_length) + 1
    print(f'max length found: {MAX_LENGTH}\nData Prep: Done')

    # Training
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder0 = DecoderRNN(hidden_size, MAX_LENGTH).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)

    model = Seq2Seq(encoder1, attn_decoder1, MAX_LENGTH, pairs, input_lang, output_lang)
    num_iters = 15000 # [75000 ~ 1.15 hrs]
    model.trainIters(num_iters, print_every=5000)
    print("Train: Done")

    # Evaluation
    print("\nEvaluating training set:")
    model.evaluateSet(pairs)
    print("\nEvaluating test set:")
    model.evaluateSet(parse_tasks_into_pairs(result, testing_tasks))

    print("Enter a positive integer to evaluate a random number of sentences or a sentence to evaluate.")
    print("(Enter 'q' or 'quit' to quit.)")
    more = input(">> ")
    while more != "q" and more != "quit":
        if more.isdigit():
            model.evaluateRandomly(n=int(more))
        else:
            model.evaluate(more)
        more = input(">> ")
    print("\nTest: Done")

    sys.exit(0)


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
# -----------------------------------------------------------------------------

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.max_sentence_length = 0 

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
        
        sent_length = len(sentence.split(' '))
        if sent_length > self.max_sentence_length:
            self.max_sentence_length = sent_length

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def indexesFromSentence(self, sentence):
        return [self.word2index[word] for word in sentence.split(' ')]
    
    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

class BERTLang:
    def __init__(self, name, tokenizer):
        self.name = name
        self.tokenizer = tokenizer
        self.max_sentence_length = 0 
        self.n_words = 2  # Count SOS and EOS
        self.bert_token_set = set()
        self.word_set = set()

    def addSentence(self, sentence):
        indices = self.bert_encoding(sentence)
        for token in indices:
            self.bert_token_set.add(token)
        self.n_words = len(self.bert_token_set)
        if len(indices) > self.max_sentence_length:
            print("new largest")
            print(f'{sentence} - {len(sentence)}')
            print(f'{indices} - {len(indices)}')
        self.max_sentence_length = max(self.max_sentence_length, len(indices))

    def bert_encoding(self, sentence):
        return self.tokenizer.encode(sentence, add_special_tokens=False) #, return_tensors='pt')
    
    def tensorFromSentence(self, sentence):
        indexes = self.bert_encoding(sentence)
        # indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
        
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Seq2Seq:
    def __init__(self, encoder, decoder, max_length, train_pairs, input_lang, output_lang):
        self.encoder = encoder
        self.decoder = decoder 
        self.max_length = max_length
        self.train_pairs = train_pairs
        self.input_lang = input_lang
        self.output_lang = output_lang 

        # for training w a teacher
        self.teacher_forcing_ratio = 0.5

    # Helper Functions
    def tensorsFromPair(self, pair):
        input_tensor = self.input_lang.tensorFromSentence(pair[0])
        target_tensor = self.output_lang.tensorFromSentence(pair[1])
        return (input_tensor, target_tensor)

    def train(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion):
        encoder_hidden = self.encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        
        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def evaluate(self, sentence):
        with torch.no_grad():
            input_tensor = self.input_lang.tensorFromSentence(sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def evaluateRandomly(self, n=10):
        for i in range(n):
            pair = random.choice(self.train_pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')
    
    def evaluateSet(self, in_out_pairs, verbose=False):
        total_correct = 0
        for pair in in_out_pairs:
            output_words, attentions = self.evaluate(pair[0])
            output_sentence = ' '.join(output_words)
            if verbose: print(f'> {pair[0]} --> {output_sentence} [{pair[1]}]')
            if output_sentence == pair[1][0:-1]: # removes EOS tag
                total_correct += 1
        print(f'Total perfectly recovered: {total_correct}/{len(in_out_pairs)} ({1.0 * total_correct/len(in_out_pairs)})')


    def trainIters(self, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        training_pairs = [self.tensorsFromPair(random.choice(self.train_pairs)) for i in range(n_iters)]
        criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train(input_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)

# "This is a helper function to print time elapsed and estimated time remaining given the current time and progress %."
import math
import time

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
# -----------------------------------------------------------------------------
