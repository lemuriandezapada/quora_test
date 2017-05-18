import torch
import torch.autograd as ta
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

#We're going to use a form of residual blocks for our passes
#https://arxiv.org/abs/1512.03385
#http://icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf
#The skip connections help avoid the vanishing gradient problem
#Dropout and batchnorm try to help avoid overfitting
#Didn't have enough time though to find the right setting


class Residual(nn.Module):
    def __init__(self,insize, outsize):
        super(Residual, self).__init__()
        drate = .3
        self.math = nn.Sequential(
                                 nn.BatchNorm1d(insize),
                                 nn.Dropout(drate),
                                 nn.Linear(insize, outsize),
                                 nn.PReLU(),
                                )
        self.skip = nn.Linear(insize, outsize)
        
    def forward(self, x):
        return self.math(x)+self.skip(x)
        

#Exactly the same as above, except for convolutional layers
#Convolution size is 3, since we're really not in precise renditions of higher order n-grams larger than 3
#The skip connection of size 1 lets lower level features "seep" through unperturbed

class ConvRes(nn.Module):
    def __init__(self,insize, outsize):
        super(ConvRes, self).__init__()
        drate = .3
        self.math = nn.Sequential(
                                 nn.BatchNorm1d(insize),
                                 nn.Dropout(drate),
                                 nn.Conv1d(insize, outsize,3, padding=1),
                                 nn.PReLU(),
                                )
        self.skip = nn.Conv1d(insize, outsize,1, padding=0)
        
    def forward(self, x):
        return self.math(x)+self.skip(x)
        

#Let's define a LSTM class
#This takes a 400d input vector (300d from pretrained glove embeddings, 50d from tunable word embeddings, and 50d from out of glove vocabulary word embeddings)
#
class WordLSTM(nn.Module):
    def __init__(self, rsize = 256):
        super(WordLSTM, self).__init__()
        #typical Pytorch cudnn LSTM declaration
        self.rnn = nn.LSTM(input_size=400, 
                           hidden_size=rsize, 
                           batch_first=True, 
                          )

        #readout function of the reservoir outputs to relieve some of the nonlinear pressure off the reservoir
        self.readout = nn.Sequential(
                                     Residual(rsize, 128),
                                     Residual(128, 128),
                                     Residual(128, 128),
                                    )
        
        
    def forward(self,x, h_0):
        #pass the embedding input sequence through the LSTM
        #h_0 is stand in for the initial hidden states, which is always 0 in this case
        fseq, (h_n, c_n) = self.rnn(x, (h_0, h_0))
        
        #there are several ways we can pool this, but just taking the last hidden state works fine enough for such a problem
        x = fseq[:,-1,:]
        
        #read out the final state and return it
        x = self.readout(x)
        return x
        
        
#Let's define the character convnet
#No time to run an actual charLSTM, also we don't have enough data for it to work properly        
class CharCNN(nn.Module):
    def __init__(self, charembsize):
        super(CharCNN, self).__init__()
        #take character indices and output dense encodings
        #index #0 always outputs 0
        self.embedding = nn.Embedding(charembsize,100, padding_idx=0)
        #3 layers of just full convolutions, no pooling
        #we don't preprocess the inputs so we don't want to accidentally mess up dimensionality
        #activation functions take care of the nonlinearity
        self.cnn = nn.Sequential(ConvRes(100, 128),
                                 ConvRes(128, 128),
                                 ConvRes(128, 128),
                                )
                

        #some more transforms after the temporal pooling
        self.readout = nn.Sequential(Residual(128, 128),
                                     Residual(128, 128),
                                    )
        
        
    def forward(self,x):
        #make embeddings
        x = self.embedding(x)
        
        #shift dimensions to fit pytorch's dimension ordering rules for convs
        x = torch.transpose(x,1,2)
        
        #get the output and pool
        x = self.cnn(x)
        x = torch.mean(x,2)
        
        #remove a dimension and feed through the readout
        x = x.view(x.size(0),x.size(1))
        x = self.readout(x)
        return x
        
        
        
        
        
        
        
#A siamese network takes two different sets of data, processes them through two identical branches
#deploys some fancy merging function and minimizes a final metric between the two.
#Similar samples should have a low metric, dissimilar samples should have a large metric.
#Check out my (one and only) paper on this exact topic:
#https://www.researchgate.net/publication/304834009_Learning_Text_Similarity_with_Siamese_Recurrent_Networks
#although in this case the metric is a probability :)
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.word_lstm_encoder = WordLSTM()
        self.char_cnn_encoder = CharCNN()
        #The final classifier is a typical feedforward NN with one categorical output
        self.classifier = nn.Sequential(
                            Residual(4*(128), 256),
                            Residual(256, 128),
                            Residual(128, 128),
                            nn.Linear(128, 1),
                            nn.Sigmoid()
                            )
    def forward(self, x, y, cx, cy, h_0):
        
        #we take two identical branches of parameters and feed them two different datas
        #first on the word lstm branches
        x = self.word_lstm_encoder(x, h_0)
        y = self.word_lstm_encoder(y, h_0)
        
        #then on the character conv branches
        cx = self.char_cnn_encoder(cx)
        cy = self.char_cnn_encoder(cy)
        
        #compute a distance metric. This can be tricky and needs experimenting for different tasks.
        #Here we use two: a product between the embeddings and an euclidean distance
        #And do it separately for words and characters
        #You can also, in principle, just concatenate the outputs, but the branches together
        #but then you need to compute them twice (also feed y's through the x branch and x's through the y branch
        #to ensure the readout (classifier) comes out symmetrical and the sentences transitve
        
        z = torch.cat([x*y, (x-y)**2, cx*cy, (cx-cy)**2], 1)
        #finally output the probability that these two questions mean the same thing
        z = self.classifier(z)
        return z
