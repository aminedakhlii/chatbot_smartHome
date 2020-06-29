import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle


with open("data.json") as file:
    data = json.load(file)
print(data)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


def updateModel():
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    updateModel()
#run the code below only if you want to update the model


print(labels)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(instruction):
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = instruction
        if inp.lower() == "quit":
            break
        if inp.lower() == "lilia":
            return random.choice(['yes amigo !','how can I help','tell me..'])

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]



        if results[0][results_index] > 0.8:

            if tag == 'order':

                if 'open' in inp and 'garage' in inp:
                    with open('commands.txt' , 'w') as f:
                        f.write('python open.py')
                    res = 'openinig garage ...'
                if 'open' in inp and 'home' in inp:
                    with open('commands.txt' , 'w') as f:
                        f.write('python openH.py')

                if 'close' in inp and 'garage' in inp:
                    with open('commands.txt' , 'w') as f:
                        f.write('python close.py')

                if 'close' in inp and 'home' in inp:
                    with open('commands.txt' , 'w') as f:
                        f.write('python closeH.py')

            if tag == 'status':
                if 'garage' in inp:
                    pass
                if 'home' in inp:
                    pass
                if 'light' in inp:
                    pass

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

                    res = 'lilia :'+random.choice(responses)
                    print(res)

        elif results[0][results_index] > 0.7 and results[0][results_index] < 0.8:

                with open("potentialdata.txt", "a") as pdata:
                    pdata.write('doubted as ' + tag + ' : ' + instruction + '\n')

                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                        res = 'lilia : I am not very sure about what to say mmm ...' + random.choice(responses)
                        print(res)
        else:
            with open("potentialdata.txt" , "a") as pdata:
                pdata.write('not figured : ' + instruction + '\n')
            res = 'lilia : sorry I didnt get what you are trying to say ... I am still learning'
            print(res)
        return res
