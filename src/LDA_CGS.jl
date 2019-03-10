module LDA_CGS

export LDA, initialize_alpha, initialize_beta, initialize_topic, MCMC,
    update_prior, PPL, sample_Nkv, sample_Ndk, run_LDA

using SpecialFunctions

mutable struct LDA
    #=
    We assume D is the number of training documents
    =#
    K::Int64
    V::Int64
    D::Int64
    Nk::Array{Int64,1}
    Nkv::Array{Int64,2}
    Ndk::Array{Int64,2}
    Nd::Array{Int64,1}
    nonzeroNdkindex::Array{Array{Int64,1},1}
    nonzeroNkvindex::Array{Array{Int64,1},1}
    z::Array{Array{Array{Int64,1},1},1} #z[d][iv] = [word_id, k_1, k_2, ...]
    alpha::Array{Float64,1}
    beta::Float64
    s::Float64
    r::Float64
    c::Array{Float64,1}
    S::Int64 #Sampling number
    PPL::Float64
    pdv::Array{Array{Float64,1},1}
    Nkv_mean::Array{Float64,2}
    Ndk_mean::Array{Float64,2}
    LDA(K, V) = new(K, V)
end

function initialize_alpha(x::LDA, alpha)
    x.alpha = alpha
    return
end

function initialize_beta(x::LDA, beta)
    x.beta = beta
    return
end

function initialize_topic(x::LDA, corpus_train)
    #=
    corpus is training corpus
    =#
    x.D, = size(corpus_train)
    K = x.K
    D = x.D
    V = x.V
    x.s = 0.0
    x.r = 0.0
    x.c = zeros(K)
    x.Nk = zeros(K)
    x.Nkv = zeros(K, V)
    x.Ndk = zeros(D, K)
    x.Nd = zeros(D)
    x.z = Array{Array{Int64,1},1}[]
    x.nonzeroNdkindex = Array{Int64,1}[]
    x.nonzeroNkvindex = Array{Int64,1}[]
    for d in 1:D
        corpus_d = corpus_train[d]
        push!(x.z, Array{Int64,1}[])
        for iv in eachindex(corpus_d)
            (v, Ndv) = corpus_d[iv]
            push!(x.z[d], [])
            for i in 1:Ndv
                k = rand(1:K)
                push!(x.z[d][iv],k)
                x.Ndk[d,k] += 1
                x.Nkv[k,v] += 1
                x.Nk[k] += 1
                x.Nd[d] += 1
            end
        end
        
        temp = Int64[]
        for k in 1:K
            if x.Ndk[d,k] != 0
                push!(temp, k)
            end
        end
        push!(x.nonzeroNdkindex, temp)
    end

    for v in 1:V
        temp = Int64[]
        for k in 1:K
            if x.Nkv[k,v] != 0
                push!(temp, k)
            end
        end
        push!(x.nonzeroNkvindex, temp)
    end
end

function subtract(x::LDA, d::Int64, v::Int64, kold::Int64)
    V = x.V
    #subtract (d,n) contributions
    x.s -= x.alpha[kold]*x.beta/(x.beta*V + x.Nk[kold])
    x.r -= x.beta*x.Ndk[d, kold]/(x.beta*V + x.Nk[kold])
    x.Nk[kold] -= 1
    x.Ndk[d, kold] -= 1
    x.Nkv[kold, v] -= 1
    x.c[kold] = (x.alpha[kold] + x.Ndk[d, kold])/(x.beta*V + x.Nk[kold])
    x.s += x.alpha[kold]*x.beta/(x.beta*V + x.Nk[kold])
    x.r += x.beta*x.Ndk[d, kold]/(x.beta*V + x.Nk[kold])
    return
end

function update(x::LDA, d::Int64, v::Int64, knew::Int64)
    V = x.V
    x.s -= x.alpha[knew]*x.beta/(x.beta*V + x.Nk[knew])
    x.r -= x.beta*x.Ndk[d, knew]/(x.beta*V + x.Nk[knew])
    x.Nk[knew] += 1
    x.Ndk[d, knew] += 1
    x.Nkv[knew, v] += 1
    x.c[knew] = (x.alpha[knew] + x.Ndk[d, knew])/(x.beta*V + x.Nk[knew])
    x.s += x.alpha[knew]*x.beta/(x.beta*V + x.Nk[knew])
    x.r += x.beta*x.Ndk[d, knew]/(x.beta*V + x.Nk[knew])
    return
end

function MCMC(x::LDA, train_corpus)
    K = x.K
    D = x.D
    V = x.V
    
    x.s = 0.0
    for k in 1:K
        x.s += x.alpha[k]*x.beta/(x.beta*V + x.Nk[k])
        x.c[k] = (x.alpha[k])/(x.beta*V + x.Nk[k])
    end
    
    for d in 1:D

        x.r = 0.0
        for k in x.nonzeroNdkindex[d]
            x.r += x.beta*x.Ndk[d,k]/(x.beta*V + x.Nk[k])
            x.c[k] = (x.alpha[k] + x.Ndk[d,k])/(x.beta*V + x.Nk[k])
        end
        
        # Iterate for each word
        words = train_corpus[d] #assume Tuple, consisting of (word_id, Number of words)
        for iv in eachindex(words)
            (v, Ndv) = words[iv]
            kolds = x.z[d][iv]

            for i in 1:Ndv
                kold = kolds[i]
                subtract(x, d, v, kold)
                
                # rearrange nonzero index for Ndk
                if x.Ndk[d, kold] == 0
                    filter!(e->e!=kold, x.nonzeroNdkindex[d])
                end
                if x.Nkv[kold, v] == 0
                    filter!(e->e!=kold, x.nonzeroNkvindex[v])
                end
                
                # Get nonzero index for Nkv
                q = 0.0
                for k in x.nonzeroNkvindex[v]
                    q += x.Nkv[k, v]*x.c[k]
                end
                
                #MCMC
                zsum = rand()*(x.s+x.r+q)
                knew = 0
                
                if zsum < q
                    j = 1
                    while zsum > 0
                        knew = x.nonzeroNkvindex[v][j]
                        zsum -= x.Nkv[knew, v]*x.c[knew]
                        j += 1
                    end
                    
                elseif zsum < x.r + q
                    j = 1
                    zsum = zsum - q
                    while zsum > 0
                        knew = x.nonzeroNdkindex[d][j]
                        zsum -= x.beta*x.Ndk[d, knew]/(x.beta*V + x.Nk[knew])
                        j += 1
                    end
                    
                elseif zsum <= x.r + q + x.s
                    zsum = zsum - (q+x.r)
                    knew = 0
                    for k in 1:K
                        zsum -= x.alpha[k]*x.beta/(x.beta*V + x.Nk[k])
                        knew = k
                        if zsum < 0
                            break
                        end
                    end
                    
                else
                    print("something unexpected happened\n")
                    print("sum=", zsum, ", r+s+q=", x.r+x.s+q,"\n")
                    print("s, r, q=", x.s, ",", x.r, ",", q, "\n")
                end

                update(x, d, v, knew)
                
                if x.Ndk[d, knew] == 1
                    push!(x.nonzeroNdkindex[d], knew)
                end
                if x.Nkv[knew, v] == 1
                    push!(x.nonzeroNkvindex[v], knew)
                end

                x.z[d][iv][i] = knew
            end
        end
        for k in x.nonzeroNdkindex[d]
            x.c[k] = (x.alpha[k])/(x.beta*V + x.Nk[k])
        end
    end
end

function update_prior(x::LDA)
    K = x.K
    V = x.V
    D = x.D
    # Update hyoper parameters
    alpha_sum = sum(x.alpha)
    
    beta_num=0.0
    beta_den=0.0
    for k in 1:K
        alpha_num = 0.0
        alpha_den = 0.0
        for v in 1:V
            beta_num += digamma(x.Nkv[k,v]+x.beta)
        end
        beta_den += digamma(x.Nk[k]+x.beta*V)
        for d in 1:D
            alpha_num +=digamma(x.Ndk[d,k]+x.alpha[k])
            alpha_den += digamma(sum(x.Ndk[d,:])+alpha_sum)
        end
        x.alpha[k] = x.alpha[k]*(alpha_num-D*digamma(x.alpha[k]))/(alpha_den-D*digamma(alpha_sum))
    end
    x.beta = x.beta*(beta_num-K*V*digamma(x.beta))/(V*beta_den-K*V*digamma(x.beta*V))
    return
end

function PPL(x::LDA, corpus_test)
    D = size(corpus_test)[1]
    K = x.K
    V = x.V
    
    alpha_sum = sum(x.alpha)
    N = sum(x.Nd)
    L = 0.0 #log likelihood
    
    if x.S == 1
        x.pdv = Array{Float64,1}[]
        for words in corpus_test
            push!(x.pdv, rand(size(words)[1]))
        end
    end
    
    for d in eachindex(corpus_test)
        words = corpus_test[d] #assume Tuple, consisting of {v:Ndv}
        for iv in eachindex(words)
            (v, Ndv) = words[iv]
            x.pdv[d][iv] = (x.S-1.0)/x.S * x.pdv[d][iv]
            for k in 1:K
                phi_kv = (x.Nkv[k, v] + x.beta)/(x.Nk[k] + x.beta*V)
                theta_dk = (x.Ndk[d,k] + x.alpha[k])/(x.Nd[d] + alpha_sum)
                x.pdv[d][iv] += phi_kv * theta_dk /x.S
            end
            L += Ndv * log(x.pdv[d][iv])
        end
    end
    x.PPL = exp(-L/N)
end

function sample_Nkv(x::LDA)
    K = x.K
    V = x.V
    
    if x.S == 1
        x.Nkv_mean = zeros(K, V)
    end
    
    for k in 1:K
        for v in 1:V
            x.Nkv_mean[k, v] = 1.0/x.S*x.Nkv[k,v] + (x.S-1)/x.S*x.Nkv_mean[k,v]
        end
    end
end

function sample_Ndk(x::LDA)
    D = x.D
    K = x.K
    V = x.V
    
    if x.S == 1
        x.Ndk_mean = zeros(D, K)
    end
    
    for d in 1:D
        for k in 1:K
            x.Ndk_mean[d, k] = 1.0/x.S*x.Ndk[d,k] + (x.S-1)/x.S*x.Ndk_mean[d,k]
        end
    end
end

function run_LDA(x::LDA, corpus_train, corpus_test, burnin=400, sample=100)
    #=
    K: Number of topics
    V: Size of vocabulary
    corpus: We assume corpus[i] = [(word_id, Number of the word in document i), (, ), ...]
    =#
    
    initialize_alpha(x, rand(x.K))
    initialize_beta(x, rand())
    initialize_topic(x, corpus_train)
    x.PPL = 1000.0
    x.S = 0

    println("Burn-in (period=$burnin)...")
    for i in 1:burnin
        MCMC(x, corpus_train)
        update_prior(x)
        println("epoch=", i)
    end

    println("Sampling from the posterior...")
    for i in 1:sample
        x.S += 1
        MCMC(x, corpus_train)
        update_prior(x)
        PPL(x, corpus_test)
        sample_Nkv(x)
        sample_Ndk(x)
        println("epoch=", i, ", PPL=", x.PPL)
    end
    end
end
