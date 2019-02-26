using Test
using JSON
using LDA_CGS

function test()
    vocab = []
    texts = []

    open("./data/hep-ph.json", "r") do f
        i=1
        for line in eachline(f)
            jdict = JSON.parse(line)
            words = split(jdict["summary"],r" | |\.|\,|\n|\$")
            push!(texts, words)
            union!(vocab, collect(words))
            i += 1
        end
    end


    
    vocab_size,  = size(vocab)
    word_to_id = Dict()
    id_to_word = Dict()
    for i in 1:vocab_size
        word_to_id[vocab[i]] = i
        id_to_word[i] = vocab[i]
    end

    texts_size, = size(texts)
    corpus = []
    for text in texts
        words_id = map(w->word_to_id[w], text)
        
        counts_d = Dict{Int, Int}()
        for iv in words_id
            counts_d[iv] = get(counts_d, iv, 0) + 1
        end
        
        counts_d_tuple = []
        for (key, ct) in counts_d
            push!(counts_d_tuple, (key, ct))
        end
        push!(corpus, counts_d_tuple)
    end


    train_ratio = 0.8
    train_corpus = []
    test_corpus = []
    
    for d in corpus
        Nd, = size(d)
        length = Int(round(Nd*0.8))
        push!(train_corpus, d[1:length])
        push!(test_corpus, d[length+1:end])
    end


    K = 20
    V = vocab_size
    x = LDA_CGS.LDA(K, V)
    
    burnin = 10
    sample = 2
    LDA_CGS.run(x, train_corpus, test_corpus, burnin, sample)

    return true
end

@test test()
