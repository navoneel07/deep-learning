def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

raw_text = load_doc('kafka.txt')
tokens = raw_text.split()
raw_text = ' '.join(tokens)

length = 10
sequences = []
for i in range(length, len(raw_text)):
    seq = raw_text[i-length:i+1]
    sequences.append(seq)

out_filename = "kafka_sequences.txt"
save_doc(sequences, out_filename)
