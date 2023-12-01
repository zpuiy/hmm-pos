import json
# from os import unlink
import numpy as np
import copy

# path to load raw data files
path_train = './data/train.json'
path_dev = './data/dev.json'
path_test = './data/test.json'
# path to save required files
path_vocab = './out/vocab.txt'
path_hmm = './out/hmm.json'
path_greedy = './out/greedy.json'
path_viterbi = './out/viterbi.json'


# load all data
with open(path_train) as f:
    train_data = json.load(f)

with open(path_dev) as f:
    dev_data = json.load(f)

with open(path_test) as f:
    test_data = json.load(f)



# Task 1: Vocabulary Creation
vocab_dict = {}

labels = set()
for item in train_data:
    sentence = item['sentence']
    labels.update(item['labels'])
    for word in sentence:
        vocab_dict[word] = vocab_dict.get(word, 0) + 1

new_dict = {}
thres = 2
# print(f'Threshold: {thres}')
unk_freq = 0
unk_words = []
for word, freq in vocab_dict.items():
    if freq < thres:
        unk_freq = unk_freq + 1
        unk_words.append(word)
    else:
        new_dict[word] = freq

# print(f'vocab size: {len(new_dict) + 1}')
# print(f'unknonwn frequency: {unk_freq}')

vocab = sorted(new_dict.items(), key=lambda x: x[1], reverse=False)
# array for all words in vocab
unique_words = np.array(list(new_dict.keys()))
unique_words = np.insert(unique_words, 0, '<unk>')

# create a dictionary to store word indices
word_index = {}
for i in range(len(unique_words)):
    word_index[unique_words[i]] = i

# array for all tags
tags = np.array(list(labels))

# create a dictionary to store word indices
tag_index = {}
for i in range(len(tags)):
    tag_index[tags[i]] = i

vocab_txt = []
for i in range(len(vocab)):
    vocab_txt.append(f'{vocab[i][0]}\t{i+1}\t{vocab[i][1]}')

with open(path_vocab, 'w') as f:
    f.write(f'<unk>\t0\t{unk_freq}\n')
    f.writelines('\n'.join(vocab_txt))

# replace unknown words
def replace_unknown(data):
    new = data.copy()
    for item in new:
        item['sentence'] = ['<unk>' if word not in word_index else word for word in item['sentence']]
    return new
        
ori_test = copy.deepcopy(test_data)
new_train = replace_unknown(train_data)
new_test = replace_unknown(test_data)
new_dev = replace_unknown(dev_data)



# task 2
# initialize transition parameter matrix
transition = np.zeros((len(labels), len(labels)))

count_null_s = np.zeros(len(labels))
count_s_s = np.zeros((len(labels), len(labels)))
count_s = np.zeros(len(labels))

# count_x_s = np.zeros()

for item in new_train:
    curr_labels = item['labels']
    count_null_s[tag_index.get(curr_labels[0])] += 1
    count_s[tag_index.get(curr_labels[0])] += 1
    for i in range(0, len(curr_labels) - 1):
        index_s = tag_index.get(curr_labels[i])
        index_s_to = tag_index.get(curr_labels[i+1])
        count_s_s[index_s][index_s_to] += 1
        count_s[index_s_to] += 1

# count_s = np.sum(count_s_s, axis=1)
transition = count_s_s / count_s[:, np.newaxis]
prior = count_null_s / len(new_train)

# create emission matrix
emission = np.zeros((len(labels), len(unique_words)))

count_x_s = np.zeros((len(labels), len(unique_words)))
count_s = np.zeros(len(labels))

for item in new_train:
    curr_labels = item['labels']
    curr_sen = item['sentence']
    for i in range(len(curr_labels)):
        index_s = tag_index.get(curr_labels[i])
        count_s[index_s] += 1
        index_x = word_index.get(curr_sen[i])
        count_x_s[index_s][index_x] = count_x_s[index_s][index_x] + 1

emission = count_x_s / count_s[:, np.newaxis]

# print(f'transition size: {len(tags) * len(tags)}')
# print(f'emission size: {len(tags) * len(unique_words)}')

# create output file
initial_dict = {}
for i in range(len(tags)):
    initial_dict[tags[i]] = prior[i]

transition_dict = {}
for i in range(len(transition)):
    curr = transition[i]
    for j in range(len(curr)):
        transition_dict[f'({tags[i]}, {tags[j]})'] = transition[i][j]

emission_dict = {}
for i in range(len(emission)):
    curr = emission[i]
    for j in range(len(curr)):
        emission_dict[f'({tags[i]}, {unique_words[j]})'] = emission[i][j]

hmm = {"initial": initial_dict, "transition": transition_dict, "emission": emission_dict}

# write learned model into a model file in json format, named hmm.json.
with open(path_hmm, 'w') as f:
    json.dump(hmm, f)



# task 3
# greedy decoding
def greedy(sentence):
    result = []
    w_index = word_index.get(sentence[0]) 
    prev_y = np.argmax(prior * emission[:, w_index])
    result.append(tags[prev_y])
    for i in range(1, len(sentence)) :
        w_index = word_index.get(sentence[i])
        curr_y = np.argmax(transition[prev_y] * emission[:,w_index])
        prev_y = curr_y
        result.append(tags[curr_y])
    return result

# create a function to calculate accuracy
def evaluate(model):
    n_true = 0
    n_total = 0
    for item in new_dev:
        curr_sen = item['sentence']
        curr_pred = model(curr_sen)
        true_pred = item['labels']
        n_total += len(curr_sen)
        for i in range(len(curr_sen)):
            if true_pred[i] == curr_pred[i]:
                n_true += 1

    return n_true / n_total

# evaluate it on the development data
greedy_accuracy = evaluate(greedy)
# print(f'Accuracy for greedy decoding: {greedy_accuracy}')

# make prediction on test data
def predict(model):
    result = copy.deepcopy(ori_test)
    for i in range(len(new_test)):
        curr_sen = new_test[i]['sentence']
        result[i]['labels'] = model(curr_sen)
    return result

with open(path_greedy, 'w') as f:
        json.dump(predict(greedy), f)



# task 4
def viterbi(sentence):
    T1 = np.zeros((len(tags), len(sentence)), dtype=float)
    T2 = np.zeros((len(tags), len(sentence)), dtype=int)
    for s in range(len(tags)):
        index_x = word_index.get(sentence[0])
        T1[s][0] = prior[s] * emission[s][index_x]

    for o in range(1, len(sentence)):
        for s in range(len(tags)):
            index_x = word_index.get(sentence[o])
            k = np.argmax(T1[:,o-1]*transition[:,s]*emission[s,index_x])
            T1[s,o] = T1[k][o-1]*transition[k,s]*emission[s,index_x]
            T2[s,o] = k

    best_path = []
    k = np.argmax(T1[:,len(sentence) - 1])
    for o in range(len(sentence) -1, -1, -1):
        best_path.insert(0,tags[k])
        k = T2[k,o]
    return best_path


# evaluate it on the development data
viterbi_accuracy = evaluate(viterbi)
# print(f'Accuracy for viterbi decoding: {viterbi_accuracy}')

# make prediction on test data
with open(path_viterbi, 'w') as f:
        json.dump(predict(viterbi), f)