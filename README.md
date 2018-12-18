# LDA for Julia
Perform Collapsed Gibbs Sampling of Latent Dirichlet Allocation.

## Topic model
Latent Dirichlet Allocation (LDA) is the Bayesian method to analyze discrete data such as the set of documents, originally proposed in [[Blei et. al., 2003]](https://dl.acm.org/citation.cfm?id=944919.944937).
LDA infers a latent topic of each word in every document.

Here I implement the collaped Gibbs sampling (CGS) method to calculate posterior topic distributions.
The efficient method proposed in [[Yao et. al., 2009]](https://dl.acm.org/citation.cfm?doid=1557019.1557121) is adopted.

## Input data structure

The collection of documents, we call corpus, is assumed to take the form of D-dimensional array whose elements take the form
```
corpus[d][i] = (v, Ndv)
```
where 
```
d = 1, ..., D (the number of the documents)
i: refers to the i-th word in d-th document
v: ID of the i-th word
Ndv: number of the word v appearing in d-th document
```

## Simple run

Once you prepare corpus, you can analyze the topic by
```
include("LDA_CGS.jl")
x = LDA_CGS.LDA(K, V)
LDA_CGS.run(x, train_corpus, test_corpus, burnin, sample)
```
The parameters are
```
K: number of topics
V: the size of the vocanulary
train_corpus: corpus for training the model
test_corpus: corpus for evaluation of perplexity
burnin: burn-in period of MCMC, during which we update the posterior but do not sample from it
sample: number of sampling
```

After run, you can check:

-```x.PPL```: perplexity of the model (smaller perplexity is better).

-```x.Nkv_mean```: word distribution of each topic; (k,v) element is mean number of word v in topic k after MCMC sampling.

-```x.Ndk_mean```: topic distribution of each document; (d,k) element is mean number of topic k in document d after MCMC sampling.

See also ```test/test_run.ipynb``` where I analyze the topic of abstracts of ```arxiv/hep-ph``` papers.
