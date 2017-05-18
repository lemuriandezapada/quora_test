import numpy as np
#Function to convert pairs of "parsed" words into sequences of tokens. 
#Token #0 yields a fully 0 embedding. Great for padding, as its embedding will always output all 0's and will never train
#we also pad every sentence to the length of our maximum sentence in the current batch
#This gives us variety in padding length for question pairs, making it harder for the network to 
#Try to do question matching based on length. We really want it to use actual salient sentence information
def generate_inseqs(embsize, inset, leftoverinset, charinset):
    #Make the token index sequences for the glove WVs
    inseq0 = [[x if x<embsize else 0 for x in sent[0]] for sent in inset]
    maxlen = max([len(x) for x in inseq0])
    inseq0 = np.array([[0]*(maxlen-len(x))+x for x in inseq0])
    inseq1 = [[x if x<embsize else 0 for x in sent[1]] for sent in inset]
    maxlen = max([len(x) for x in inseq1])
    inseq1 = np.array([[0]*(maxlen-len(x))+x for x in inseq1])

    #Make the token index sequences for the leftover WVs
    linseq0 = [[x for x in sent[0]] for sent in leftoverinset]
    maxlen = max([len(x) for x in linseq0])
    linseq0 = np.array([[0]*(maxlen-len(x))+x for x in linseq0])
    linseq1 = [[x for x in sent[1]] for sent in leftoverinset]
    maxlen = max([len(x) for x in linseq1])
    linseq1 = np.array([[0]*(maxlen-len(x))+x for x in linseq1])

    #Make the token index sequences for the characters
    cinseq0 = [[x for x in sent[0]] for sent in charinset]
    maxlen = max([len(x) for x in cinseq0])
    cinseq0 = np.array([[0]*(maxlen-len(x))+x for x in cinseq0])
    cinseq1 = [[x for x in sent[1]] for sent in charinset]
    maxlen = max([len(x) for x in cinseq1])
    cinseq1 = np.array([[0]*(maxlen-len(x))+x for x in cinseq1])
    
    #return them all
    return inseq0, inseq1, linseq0, linseq1, cinseq0, cinseq1
    
    
#function to add noise 
def addnoise(seq, noiserate = 0.1):
    return seq * (np.random.random(seq.shape) > noiserate)
