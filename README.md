# Quora Drama
Quick Pytorch attempt at the quora kaggle challenge

[![N|Solid](https://imgs.xkcd.com/comics/machine_learning.png)](https://xkcd.com/)

If you actually want to run this you'll need at least:
 - http://pytorch.org/  - python3.5 cuda version
 - https://spacy.io/ - with the english  model: https://spacy.io/docs/usage/models
 - a big GPU, if you use the whole embedding option something with at least 6GB, maybe 8.
 - 15GB of free RAM because it will pile up
Not the most memory efficient but it fits on my laptop.

Alrighty, so we're in Kaggle under the handle Praise Kek
(https://www.kaggle.com/praisekek)

As of submitting this, we're sadly only on spot 577/2890 with a LB score of 0.297 for an esnemble of 5 intermediate solutions from 2 models. Best score for a 2 model 2 solution ensemble was ~0.302. Best score for single models was 0.315.

I dug a bit for basic analysis that's not reported here, but luckily the internet already did a lot of work in presenting it:
 - https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb
 - https://www.kaggle.com/c/quora-question-pairs/discussion/31179
 - https://www.kaggle.com/c/quora-question-pairs/discussion/32819
 - https://www.kaggle.com/selfishgene/shallow-benchmark-0-31675-lb
 
I found nothing particularly surprising, other than being surprised at how many Indians are worried about currency denomination. :)

Popular methods seem to be XGBoost, and A LOT of feature engineering. Some slightly more data leak-y than others (extracting some features from the testing set). 
- https://www.kaggle.com/jturkewitz/magic-features-0-03-gain

Since we're looking for something slightly more original, let's skip all that. The whole point of ML is to avoid doing the work and let the machine work for you. So we're settling on a neural net [ensemble] solution. Although not even this is extremely original, as there are several solutions out there doing DL too on kind of the same idea, including Quora themselves :):
 - https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
 - https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur
 - https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
 - http://www.erogol.com/duplicate-question-detection-deep-learning/
 - http://sergeiturukin.com/2017/04/20/kaggle-quora-part3.html

# Alright, so what's the same as other implementations?

Dual branch Siamese network, with:
 - Monodirectional LSTMs
 - Glove word vectors pretrained by Spacy on a large corpus as inputs
 - dual euclidean/cosine distance merge nodes
 - Minimizing the log loss directly
 
# Boo! So what's different then?
 - Resnet readouts
 - Convolutional letter feature extractors (in resnet form)
 - Embeddings for new, unknown words
 - Partially trainable embeddings for partial domain adaptation
 - Input dropout
 - optimizer/data split same model ensemble [still running, not yet done]
 - No extra features. Only data. No computing lengths, frequencies, TF-IDF, one-hots, or any mambo-jumbo a linear model would like

Basically our architecture looks something like this:
[![network](https://github.com/lemuriandezapada/quora_test/blob/master/images/network.png?raw=true)]
- Words get tokenized with whatever Spacy outputs and given 3 embeddings
    - One 300d fixed embedding from Spacy's glove [if available]
    - One 50d (learnable) embedding from us [for the ones with glove vectors]
    - One 50d (learnable embedding) from us for words outside of glove vocabulary
- They then go into an LSTM and get read out on the last activation timestep
- They get read out with a resnet structure and two distance metrics are computed with the output of the second branch
- The strings also get split by most common 100 characters, and each character is given an embedding
- The embeddings then form the basis for inputs into a fully convolutional stack meant to extract some 1-7grams 
- Average pooling at the end
- and get two metrics computed
- The metrice get concatenated and read out resnet style until the final classification sigmoid
- That is all :)

# Meh, ok, so what works, what doesn't?
 - A simple MLP from averaged glove vectors already yields a LB score of 0.46. If you score lower (higher) than this, you're not actually trying. 
 - Concatenating the features instead of euclidean+product distances doesn't have any clear impact, but increases computation time.
 - Go weight decay!
 - Some drop-input helps prevent overfitting but too much will kill feature saliency.
 - CharRNN would need more data
 - Train and test data come from different distributions. Whether that needs to be corected or not, is up for debate. 
 - Bidirectional is overkill
 - Learning the distance metrics directly, or attempting a supervised/unsupervised LM hybrid doesn't provide much improvement on this data. Or at least did not in some previous incarnation of my implementation.
 - More embeddings for the same tokens are good. Some fixed, some not, and with different capacity. You want to split some words from the clusters they fell in when trained up on the 6B words set, but not so much that they loose touch with their basic meaning and fail to generalize.
 - Model averaging helps. Preferably models that are different. At least trained on different data, preferably different optimizer.
 - Honestly even just more training would help. Validation error still decreasing and intermediate submissions still boost LB ranking. But I can cook eggs on my laptop now.
 - Even more honestly, though, while it may not rank all that high (yet), this is already a more robust and generalizable solution beyond this test set, than what some other feature-based methods provide. The (non)features used here are non-specific to this particular problem domain.

