import numpy as np

data = open('kafka.txt', 'r').read()

chars = list(set(data))
vocab_size, data_size = len(chars), len(data)
#print(f"Data has {vocab_size} unique and {data_size} total characters.")
#make two dictionaries to decode and encode the chars as vectors.
char_to_int = {ch:i for i,ch in enumerate(chars)}
int_to_char = {i:ch for i, ch in enumerate(chars)}

#hyper params

hidden_size = 100
seq_length = 25
learning_rate = 1e-1

#model params

W_xh = np.random.randn(hidden_size, vocab_size) * 0.01
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
W_hy = np.random.randn(vocab_size, hidden_size) * 0.01
b_h = np.zeros((hidden_size, 1))
b_y = np.zeros((vocab_size, 1))

def lossFun(inputs, targets, hprev):
    xs, ys, hs, ps = {}, {}, {}, {}

    hs[-1] = np.copy(hprev)
    loss = 0
    #forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeroes((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(W_hh, hs[t-1]) + np.dot(W_xh, xs[t]) + b_h)
        ys[t] = np.dot(W_hy, hs[t]) + b_y
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])

    #backward pass
    dW_xh, dW_hh, dW_hy = np.zeroes_like(W_xh), np.zeroes_like(W_hh), np.zeroes_like(W_hy)
    db_h, db_y = np.zeroes_like(b_h), np.zeroes_like(b_y)
    dhnext = 
