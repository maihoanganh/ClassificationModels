function test_bivariate_Christoffel(data)
    include(data*"/plot/bivariate_data.jl");
    Lambda1=christoffel_func(N,Y1,t,d);

    p_approx1(x1,x2)=Lambda1([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    contour(x1s, x2s, p_approx1,fill=true,title = "estimation 1")

    scatter!(Y1[:,1], Y1[:,2],label ="sample 1",color="Green")
    scatter!(Y2[:,1], Y2[:,2],label ="sample 2",color="Red")
    
end

function test_bivariate_Christoffel2(data)
    include(data*"/plot/bivariate_data.jl");
    
    Lambda2=christoffel_func(N,Y2,t,d);
    
    p_approx2(x1,x2)=Lambda2([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    contour(x1s, x2s, p_approx2,fill=true,title = "estimation 2")

    scatter!(Y1[:,1], Y1[:,2],label ="sample 1",color="Green")
    scatter!(Y2[:,1], Y2[:,2],label ="sample 2",color="Red")
    
end

function test_bivariate_MLE(data)
    include(data*"/plot/bivariate_data.jl");
    
    R=1

    x1=solve_opt(N,Y1,t,R,d,delta=1,s=1,rho=1,numiter=1e5,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    
    eval_PDF1=func_eval_PDF(x1,N,d,R,ball_cons=true)
    
    p_approx1(x1,x2)=eval_PDF1([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    contour(x1s, x2s, p_approx1,fill=true,title = "estimation 1")

    scatter!(Y1[:,1], Y1[:,2],label ="sample 1",color="Blue")
    scatter!(Y2[:,1], Y2[:,2],label ="sample 2",color="Red")
end


function test_bivariate_MLE2(data)
    include(data*"/plot/bivariate_data.jl");
    
    R=1

    
    x2=solve_opt(N,Y2,t,R,d,delta=1,s=1,rho=1,numiter=1e5,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    
    eval_PDF2=func_eval_PDF(x2,N,d,R,ball_cons=true)
    
    p_approx2(x1,x2)=eval_PDF2([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    contour(x1s, x2s, p_approx2,fill=true,title = "estimation 2")

    scatter!(Y1[:,1], Y1[:,2],label ="sample 1",color="Blue")
    scatter!(Y2[:,1], Y2[:,2],label ="sample 2",color="Red")
end

function test_bivariate_MLE_density(data)
    d=2
    include(data*"/plot/bivariate_MLE_density_data.jl");
    R=1

    x=solve_opt(N,Y,t,R,d,delta=1,s=1,rho=1,numiter=1e5,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    eval_PDF=func_eval_PDF(x,N,d,R,ball_cons=true)
    
    p_approx(x1,x2)=eval_PDF([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    contour(x1s, x2s, p_approx,fill=true,title = "d = $(d)")

    scatter!(Y[:,1], Y[:,2],label ="sample")
end
    

function test_bivariate_MLE_density2(data)
    include(data*"/plot/bivariate_MLE_density_data.jl");
    d=5
    
    R=1

    x=solve_opt(N,Y,t,R,d,delta=1,s=1,rho=1,numiter=2e5,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    eval_PDF=func_eval_PDF(x,N,d,R,ball_cons=true)
    
    p_approx(x1,x2)=eval_PDF([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    contour(x1s, x2s, p_approx,fill=true,title = "d = $(d)")

    scatter!(Y[:,1], Y[:,2],label ="sample")
    
end


function test_test()
    N=2 # number of attributes
    s=2 # number of classes
    t=Vector{Int64}(undef,s) # sample sizes for traint set
    Y=Vector{Matrix{Float64}}(undef,s) # input data
    Y_train=Vector{Matrix{Float64}}(undef,s) # traint set
    Y_test=Vector{Matrix{Float64}}(undef,s) # test set
    ratio=0.8 # ratio of train set to test set

    for k=1:s 
        # take random samples
        Y[k]=Matrix{Float64}(undef,20,N)
        for j=1:20
            randx=2*rand(Float64,N).-1
            randx=0.5*rand(Float64)*randx./sqrt(sum(randx.^2))
            Y[k][j,:]=randx+(2*k-3)*[0.25;0]
        end

        t[k]=ceil(Int64,ratio*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end
    
    
    println("number of classes: ",s)
    println("ratio of train set to test set: ",ratio)
    
    

    d=Vector{Int64}(undef,s) # degrees of polynomial estimations
    R=Vector{Float64}(undef,s) # radius of the ball centered at origin containing the samples

    x=Vector{Vector{Float64}}(undef,s) # coefficients of polynomial estimations

    for k=1:s
        println("Class ",k)
        println()
        d[k]=1
        R[k]=1

        # train a model
        x[k]=solve_opt(N,Y_train[k],t[k],R[k],d[k];
                                        delta=1,s=1,rho=1,numiter=1e3,
                                        eps=1e-3,tol_eig=1e-3,
                                        ball_cons=false,feas_start=false) # Maximum likelihood estimation
        println()
        println("------------")
        println()
    end
    

    eval_PDF=Vector{Function}(undef,s) # evaluate polynomial estimations

    for k=1:s
        eval_PDF[k]=func_eval_PDF(x[k],N,d[k],R[k],ball_cons=false)
    end

    function classifier(y)
        return findmax([eval_PDF[k](y) for k=1:s])[2]
    end

    predict=Vector{Vector{Int64}}(undef,s) # prediction
    numcor=Vector{Int64}(undef,s) # number of corrections

    for k=1:s
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end


    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:s))

    println("Accuracy on test set of MLE: ",accuracy)

    println()
    println("==========================")
    println()

    Lambda=Vector{Function}(undef,s) # Christoffel function

    for k=1:s
        println("Class ",k)
        println()
        d[k]=1
        R[k]=1

        # train a model
        Lambda[k]=christoffel_func(N,Y_train[k],t[k],d[k],eps=0.0)
        println()
        println("------------")
        println()
    end

    classifier2(y)=findmax([Lambda[k](y) for k=1:s])[2]

    predict=Vector{Vector{Int64}}(undef,s) # prediction
    numcor=Vector{Int64}(undef,s) # number of corrections

    for k=1:s
        predict[k]=[classifier2(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end


    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:s))

    println("Accuracy on test set of Christoffel function: ",accuracy)
end

function test_Iris_MLE(data)
    
    df = CSV.read(data*"/Iris.csv", DataFrame)
    nr,nc=size(df)
    nc-=1
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=2:5
            D[j,i-1]=df[j,i]
        end
        if df[j,6]=="Iris-setosa"
            D[j,5]=1
        elseif df[j,6]=="Iris-versicolor"
            D[j,5]=2
        else
            D[j,5]=3
        end
    end
    
    max_col=[maximum(D[:,j]) for j=1:4]
    
    ind_zero=Vector{Int64}([])
    for j=1:4
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
    ind_zero
    
    D=D[:,setdiff(1:5,ind_zero)]
    
    D[:,1:4].-=0.5
    D[:,1:4]*=2
    max_norm_col=maximum(norm(D[j,1:4]) for j=1:nr) 
    
    D[:,1:4]/=max_norm_col
    
    Y=Vector{Matrix{Float64}}(undef,3)

    for k in 1:3
        Y[k]=D[findall(u -> u == k, D[:,end]),1:4]
    end
    N=4
    
    t=Vector{Int64}(undef,3)
    Y_train=Vector{Matrix{Float64}}(undef,3)

    for k=1:3
        t[k]=ceil(Int64,0.8*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
    end
    
    println("number of classes: ",3)
    println("ratio of train set to test set: ",0.8)
    
    Y_test=Vector{Matrix{Float64}}(undef,3)

    for k=1:3
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end
    
    d=Vector{Int64}(undef,3)
    for k=1:3
        d[k]=2
    end
    
    R=Vector{Float64}(undef,3)
    x=Vector{Vector{Float64}}(undef,3)

    @time begin
    for k=1:3
        println("Class ",k)
        println()
        R[k]=1
        x[k]=solve_opt(N,Y_train[k],t[k],R[k],d[k];delta=1,s=1,rho=1,
                             numiter=50,eps=-1e-3,tol_eig=1e-3,ball_cons=true,feas_start=false);
        println()
        println("------------")
        println()
    end
    end
    
    eval_PDF=Vector{Function}(undef,3)

    for k=1:3
        eval_PDF[k]=func_eval_PDF(x[k],N,d[k],R[k],ball_cons=true)
    end
    
    function classifier(y)
        return findmax([eval_PDF[k](y) for k=1:3])[2]
    end
    
    predict=Vector{Vector{Int64}}(undef,3)

    for k=1:3
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    end
    numcor=Vector{Int64}(undef,3)

    for k=1:3
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end
    
    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:3))
    
    println("Accuracy: ",accuracy)
    
end



function test_optdigits_Christoffel(data)
    df_train = readdlm(data*"/optdigits.tra", ',')
    df_test = readdlm(data*"/optdigits.tra", ',')
    df=[df_train;df_test]
    
    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=1:65
            D[j,i]=df[j,i]
        end

        D[j,65]+=1
    end
    
    max_col=[maximum(D[:,j]) for j=1:64]
    
    ind_zero=Vector{Int64}([])
    for j=1:64
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
    
    D=D[:,setdiff(1:65,ind_zero)]
    
    D[:,1:62].-=0.5
    D[:,1:62]*=2
    
    max_norm_col=maximum(norm(D[j,1:62]) for j=1:nr) 
    
    D[:,1:62]/=max_norm_col
    
    Y=Vector{Matrix{Float64}}(undef,10)

    for k in 1:10
        Y[k]=D[findall(u -> u == k, D[:,end]),1:62]
    end
    N=62
    
    t=Vector{Int64}(undef,10)
    Y_train=Vector{Matrix{Float64}}(undef,10)

    for k=1:10
        t[k]=ceil(Int64,0.9*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
    end
    
    println("number of classes: ",10)
    println("ratio of train set to test set: ",0.9)
    
    Y_test=Vector{Matrix{Float64}}(undef,10)

    for k=1:10
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end
    
    d=Vector{Int64}(undef,10)

    for k=1:10
        d[k]=1
        #println(binomial(N+d[k],N))
    end
    
    Lambda=Vector{Function}(undef,10)

    for k=1:10
        println("Class ",k)
        println()

        Lambda[k]=christoffel_func(N,Y_train[k],t[k],d[k],eps=0.001);
        
        println()
        println("------------")
        println()
    end
    
    function classifier(y)
        return findmax([Lambda[k](y) for k=1:10])[2]
    end
    
    predict=Vector{Vector{Int64}}(undef,10)

    for k=1:10
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    end
    
    numcor=Vector{Int64}(undef,10)

    for k=1:10
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end
    
    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:10))
    
    println("Accuracy: ",accuracy)
    
end


function test_optdigits_MLE(data)
    df_train = readdlm(data*"/optdigits.tra", ',')
    df_test = readdlm(data*"/optdigits.tra", ',')
    df=[df_train;df_test]
    
    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=1:65
            D[j,i]=df[j,i]
        end

        D[j,65]+=1
    end
    
    max_col=[maximum(D[:,j]) for j=1:64]
    
    
    ind_zero=Vector{Int64}([])
    for j=1:64
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
    
    D=D[:,setdiff(1:65,ind_zero)]
    
    D[:,1:62].-=0.5
    D[:,1:62]*=2
    
    max_norm_col=maximum(norm(D[j,1:62]) for j=1:nr) 
    
    r=1.9
    D[:,1:62]/=max_norm_col/r
    
    Y=Vector{Matrix{Float64}}(undef,10)

    for k in 1:10
        Y[k]=D[findall(u -> u == k, D[:,end]),1:62]
    end
    N=62
    
    t=Vector{Int64}(undef,10)
    Y_train=Vector{Matrix{Float64}}(undef,10)

    for k=1:10
        t[k]=ceil(Int64,0.9*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
    end
    
    println("number of classes: ",10)
    println("ratio of train set to test set: ",0.9)
    
    Y_test=Vector{Matrix{Float64}}(undef,10)

    for k=1:10
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end
    
    d=Vector{Int64}(undef,10)
    for k=1:10
        d[k]=1
        #println(binomial(N+d[k],N))
    end
    
    R=Vector{Float64}(undef,10)
    x=Vector{Vector{Float64}}(undef,10)

    @time begin
    for k=1:10
        println("Class ",k)
        println()
        R[k]=r
        x[k]=solve_opt(N,Y_train[k],t[k],R[k],d[k];delta=1,s=1,rho=1,
                             numiter=1e4,eps=1e-2,tol_eig=1e-3,ball_cons=false,feas_start=false);
            
        println()
        println("------------")
        println()
    end
    end
    
    eval_PDF=Vector{Function}(undef,10)

    for k=1:10
        eval_PDF[k]=ClassificationModels.func_eval_PDF(x[k],N,d[k],R[k],ball_cons=false)
    end
    
    function classifier(y)
        return findmax([eval_PDF[k](y) for k=1:10])[2]
    end
    
    predict=Vector{Vector{Int64}}(undef,10)

    for k=1:10
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    end
    
    numcor=Vector{Int64}(undef,10)

    for k=1:10
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end
    
    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:10))
    
    println("Accuracy: ",accuracy)
end

function test_optdigits_MLE_arb_basis(data)
    df_train = readdlm(data*"/optdigits.tra", ',')
    df_test = readdlm(data*"/optdigits.tra", ',')
    df=[df_train;df_test]
    
    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=1:65
            D[j,i]=df[j,i]
        end

        D[j,65]+=1
    end
    
    max_col=[maximum(D[:,j]) for j=1:64]
    
    ind_zero=Vector{Int64}([])
    for j=1:64
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
    
    D=D[:,setdiff(1:65,ind_zero)]
    
    D[:,1:62].-=0.5
    D[:,1:62]*=2
    
    max_norm_col=maximum(norm(D[j,1:62]) for j=1:nr) 
    r=1.9
    D[:,1:62]/=max_norm_col/r
    
    Y=Vector{Matrix{Float64}}(undef,10)

    for k in 1:10
        Y[k]=D[findall(u -> u == k, D[:,end]),1:62]
    end
    N=62
    
    t=Vector{Int64}(undef,10)
    Y_train=Vector{Matrix{Float64}}(undef,10)

    for k=1:10
        t[k]=ceil(Int64,0.7*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
    end
    
    println("number of classes: ",10)
    println("ratio of train set to test set: ",0.7)
    Y_test=Vector{Matrix{Float64}}(undef,10)

    for k=1:10
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end
    
    d=Vector{Int64}(undef,10)
    for k=1:10
        d[k]=1
    end
    
    largest_rr=binomial(2+N,N)-binomial(1+N,N)
    #println("largest_rr=",largest_rr)

    #rr=largest_rr-N:largest_rr
    rr=1:N
    
    
    R=Vector{Float64}(undef,10)
    x=Vector{Vector{Float64}}(undef,10)

    @time begin
    for k=1:10
        println("Class ",k)
        println()
        R[k]=r
        x[k]=solve_opt_arb_basis(N,Y_train[k],t[k],R[k],d[k],rr;delta=1,s=1,rho=1,
                             numiter=1e1,eps=1e-2,tol_eig=1e-3);
            
        println()
        println("------------")
        println()
    end
    end
    
    eval_PDF=Vector{Function}(undef,10)

    for k=1:10
        eval_PDF[k]=func_eval_PDF_arb_basis(x[k],N,d[k],R[k],rr)
    end
    
    function classifier(y)
        return findmax([eval_PDF[k](y) for k=1:10])[2]
    end
    
    predict=Vector{Vector{Int64}}(undef,10)

    for k=1:10
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    end
    
    numcor=Vector{Int64}(undef,10)

    for k=1:10
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end
    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:10))
    println("Accuracy: ",accuracy)
end


function test_Parkinson_Christoffel(data)
    df = CSV.read(data*"/ReplicatedAcousticFeatures-ParkinsonDatabase.csv", DataFrame)
    nr=240
    nc=46
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=4:48
            D[j,i-3]=df[j,i]
        end

        D[j,46]=df[j,3]

    end
    
    max_col=[maximum(D[:,j]) for j=1:45]
    
    for j=1:45
        D[:,j]/=max_col[j]
    end
    
    D[:,1:45].-=0.5
    D[:,1:45]*=2
    
    max_norm_col=maximum(norm(D[j,1:45]) for j=1:nr) 
    
    r=1
    D[:,1:45]/=max_norm_col/r
    
    Y1=D[findall(u -> u == 0, D[:,end]),1:45]
    N=45
    
    Y2=D[findall(u -> u == 1, D[:,end]),1:45]
    
    t1=ceil(Int64,0.9*size(Y1,1))
    Y_train1=Y1[1:t1,:]
    
    t2=ceil(Int64,0.9*size(Y2,1))
    Y_train2=Y2[1:t2,:]
    
    Y_test1=Y1[(t1+1):end,:]
    
    Y_test2=Y2[(t2+1):end,:]
    
    println("number of classes: ",2)
    println("ratio of train set to test set: ",0.9)
    
    println("Class 1")
    println()

    d1=1
    R1=1

    Lambda1=christoffel_func(N,Y_train1,t1,d1);
    println()
    println("------------")
    println()
    println("Class 2")
    println()
    d2=1
    R2=1

    Lambda2=christoffel_func(N,Y_train2,t2,d2);
    
    
    function classifier(y)
        if Lambda1(y)>Lambda2(y)
            return 1
        else
            return 2
        end
    end
    
    predict1=[classifier(Y_test1[j,:]) for j in 1:size(Y_test1,1)]
    
    numcor1=length(findall(u -> u == 1, predict1))
    
    predict2=[classifier(Y_test2[j,:]) for j in 1:size(Y_test2,1)]
    
    numcor2=length(findall(u -> u == 2, predict2))
    
    accuracy=(numcor1+numcor2)/(size(Y_test1,1)+size(Y_test2,1))
    
    println("Accuracy: ",accuracy)
    
end


function test_Parkinson_Christoffel_arb_basis(data)
    df = CSV.read(data*"/ReplicatedAcousticFeatures-ParkinsonDatabase.csv", DataFrame)
    nr=240
    nc=46
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=4:48
            D[j,i-3]=df[j,i]
        end

        D[j,46]=df[j,3]

    end
    
    max_col=[maximum(D[:,j]) for j=1:45]
    
    for j=1:45
        D[:,j]/=max_col[j]
    end
    
    D[:,1:45].-=0.5
    D[:,1:45]*=2
    
    max_norm_col=maximum(norm(D[j,1:45]) for j=1:nr) 
    
    D[:,1:45]/=max_norm_col
    
    Y1=D[findall(u -> u == 0, D[:,end]),1:45]
    N=45
    
    Y2=D[findall(u -> u == 1, D[:,end]),1:45]
    
    t1=ceil(Int64,0.9*size(Y1,1))
    Y_train1=Y1[1:t1,:]
    
    t2=ceil(Int64,0.9*size(Y2,1))
    Y_train2=Y2[1:t2,:]
    
    Y_test1=Y1[(t1+1):end,:]
    
    Y_test2=Y2[(t2+1):end,:]
    
    println("number of classes: ",2)
    println("ratio of train set to test set: ",0.9)
    
    d=1
    R=1

    largest_r=binomial(d+1+N,N)-binomial(d+N,N)
    #println("largest_r=",largest_r)

    r=1035-N:1035
    
    println("Class 1")
    println()

    d1=d
    R1=R

    Lambda1=christoffel_func_arb_basis(N,Y_train1,t1,d1,r);
    println()
    println("------------")
    println()
    println("Class 2")
    println()
    d2=d
    R2=R

    Lambda2=christoffel_func_arb_basis(N,Y_train2,t2,d2,r);
    
    function classifier(y)
        if Lambda1(y)>Lambda2(y)
            return 1
        else
            return 2
        end
    end
    
    predict1=[classifier(Y_test1[j,:]) for j in 1:size(Y_test1,1)]
    
    numcor1=length(findall(u -> u == 1, predict1))
    
    predict2=[classifier(Y_test2[j,:]) for j in 1:size(Y_test2,1)]
    
    numcor2=length(findall(u -> u == 2, predict2))
    
    accuracy=(numcor1+numcor2)/(size(Y_test1,1)+size(Y_test2,1))
    
    println("Accuracy: ",accuracy)
    
end


function test_Parkinson_MLE(data)
    df = CSV.read(data*"/ReplicatedAcousticFeatures-ParkinsonDatabase.csv", DataFrame)
    nr=240
    nc=46
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=4:48
            D[j,i-3]=df[j,i]
        end

        D[j,46]=df[j,3]

    end
    
    max_col=[maximum(D[:,j]) for j=1:45]
    
    for j=1:45
        D[:,j]/=max_col[j]
    end
    
    D[:,1:45].-=0.5
    D[:,1:45]*=2
    
    max_norm_col=maximum(norm(D[j,1:45]) for j=1:nr) 
    
    r=1.7
    D[:,1:45]/=max_norm_col/r
    
    Y1=D[findall(u -> u == 0, D[:,end]),1:45]
    N=45
    
    Y2=D[findall(u -> u == 1, D[:,end]),1:45]
    
    t1=ceil(Int64,0.9*size(Y1,1))
    Y_train1=Y1[1:t1,:]
    
    t2=ceil(Int64,0.9*size(Y2,1))
    Y_train2=Y2[1:t2,:]
    
    Y_test1=Y1[(t1+1):end,:]
    Y_test2=Y2[(t2+1):end,:]
    
    println("number of classes: ",2)
    println("ratio of train set to test set: ",0.9)
    
    println("Class 1")
    println()

    d1=1
    R1=r

    x1=solve_opt(N,Y_train1,t1,R1,d1;delta=1,s=1,rho=1,
                             numiter=1e4,eps=-1e-3,tol_eig=1e-3,ball_cons=true,feas_start=false);
    println()
    println("------------")
    println()
    println("Class 2")
    println()
    d2=1
    R2=r

    x2=solve_opt(N,Y_train2,t2,R2,d2;delta=1,s=1,rho=1,
                             numiter=1e4,eps=-1e-3,tol_eig=1e-3,ball_cons=true,feas_start=false);
    
    eval_PDF1=func_eval_PDF(x1,N,d1,R1,ball_cons=true)
    
    eval_PDF2=func_eval_PDF(x2,N,d2,R2,ball_cons=true)
    
    function classifier(y)
        if eval_PDF1(y)>eval_PDF2(y)
            return 1
        else
            return 2
        end
    end
    
    predict1=[classifier(Y_test1[j,:]) for j in 1:size(Y_test1,1)]
    
    numcor1=length(findall(u -> u == 1, predict1))
    
    predict2=[classifier(Y_test2[j,:]) for j in 1:size(Y_test2,1)]
    
    numcor2=length(findall(u -> u == 2, predict2))
    
    accuracy=(numcor1+numcor2)/(size(Y_test1,1)+size(Y_test2,1))
    
    println("Accuracy: ",accuracy)
end

function univariate_Christoffel(data)
    include(data*"/plot/univariate_Christoffel_data.jl");
    #p_exact(z)=0.5
    scatter(Y1[:,1], zeros(Float64,t),label ="sample 1",color="Blue")
    scatter!(Y2[:,1], zeros(Float64,t),label ="sample 2",color="Red")
    #plot!(p_exact, -r, r, label ="exact 1")
    
    Lambda1=christoffel_func(N,Y1,t,d);
    r=1

    p_approx1(z)=Lambda1([z;zeros(N-1)])
    plot!(p_approx1, -r, r, label = "estimation 1",#=legend=:bottomright,=#title = "t = $(t)",color="Blue")
    
    Lambda2=christoffel_func(N,Y2,t,d);
    
    p_approx2(z)=Lambda2([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "estimation 2",#=legend=:bottomright,=#title = "t = $(t)",color="Red")
end

function univariate_Christoffel2(data)
    include(data*"/plot/univariate_Christoffel_data2.jl");
    
    #p_exact(z)=0.5
    scatter(Y1[:,1], zeros(Float64,t),label ="sample 1",color="Blue")
    scatter!(Y2[:,1], zeros(Float64,t),label ="sample 2",color="Red")
    #plot!(p_exact, -r, r, label ="exact 1")
    
    Lambda1=christoffel_func(N,Y1,t,d);
    
    r=1

    p_approx1(z)=Lambda1([z;zeros(N-1)])
    plot!(p_approx1, -r, r, label = "estimation 1",#=legend=:bottomright,=#title = "t = $(t)",color="Blue")
    
    Lambda2=christoffel_func(N,Y2,t,d);
    
    p_approx2(z)=Lambda2([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "estimation 2",#=legend=:bottomright,=#title = "t = $(t)",color="Red")
end

function univariate_MLE(data)
    include(data*"/plot/univariate_MLE_data.jl");
    #p_exact(z)=0.5
    scatter(Y1[:,1], zeros(Float64,t),label ="sample 1",color="Blue")
    scatter!(Y2[:,1], zeros(Float64,t),label ="sample 2",color="Red")
    #plot!(p_exact, -r, r, label ="exact 1")
    
    R=1

    x1=solve_opt(N,Y1,t,R,d,delta=1,s=1,rho=1,numiter=1e4,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    
    eval_PDF1=func_eval_PDF(x1,N,d,R,ball_cons=true)
    
    r=1

    p_approx1(z)=eval_PDF1([z;zeros(N-1)])
    plot!(p_approx1, -r, r, label = "estimation 1",#=legend=:bottomright,=#title = "t = $(t)",color="Blue")
    
    x2=solve_opt(N,Y2,t,R,d,delta=1,s=1,rho=1,numiter=1e4,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    
    eval_PDF2=func_eval_PDF(x2,N,d,R,ball_cons=true)
    
    p_approx2(z)=eval_PDF2([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "estimation 2",#=legend=:bottomright,=#title = "t = $(t)",color="Red")
   
end


function univariate_MLE2(data)
    include(data*"/plot/univariate_MLE_data2.jl");
    #p_exact(z)=0.5
    scatter(Y1[:,1], zeros(Float64,t),label ="sample 1",color="Blue")
    scatter!(Y2[:,1], zeros(Float64,t),label ="sample 2",color="Red")
    #plot!(p_exact, -r, r, label ="exact 1")
    
    R=1

    x1=solve_opt(N,Y1,t,R,d,delta=1,s=1,rho=1,numiter=1e4,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    
    eval_PDF1=func_eval_PDF(x1,N,d,R,ball_cons=true)
    r=1

    p_approx1(z)=eval_PDF1([z;zeros(N-1)])
    plot!(p_approx1, -r, r, label = "estimation 1",#=legend=:bottomright,=#title = "t = $(t)",color="Blue")
    
    x2=solve_opt(N,Y2,t,R,d,delta=1,s=1,rho=1,numiter=1e4,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    
    eval_PDF2=func_eval_PDF(x2,N,d,R,ball_cons=true)
    
    p_approx2(z)=eval_PDF2([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "estimation 2",#=legend=:bottomright,=#title = "t = $(t)",color="Red")
end


function univariate_MLE_density(data)
    include(data*"/plot/univariate_MLE_density_data.jl");
    
    r=1
    p_exact(z)=0.5
    scatter(Y[:,1], zeros(Float64,t),label ="sample")
    plot!(p_exact, -r, r, label ="exact")
    
    d=5
    R=1

    x=solve_opt(N,Y,t,R,d,delta=1,s=1,rho=1,numiter=1e5,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    
    eval_PDF=func_eval_PDF(x,N,d,R,ball_cons=true)
    
    p_approx(z)=eval_PDF([z;zeros(N-1)])
    plot!(p_approx, -r, r, label = "d = $(d)",#=legend=:bottomright,=#title = "t = $(t)")
    
    d=10
    R=1

    x=solve_opt(N,Y,t,R,d,delta=1,s=1,rho=1,numiter=1e5,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    
    eval_PDF=func_eval_PDF(x,N,d,R,ball_cons=true)
    
    p_approx2(z)=eval_PDF([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "d = $(d)",#=legend=:bottomright,=#title = "t = $(t)")
    
    
end

function univariate_MLE_density2(data)
    include(data*"/plot/univariate_MLE_density_data2.jl");
    
    r=1
    p_exact(z)=0.5
    scatter(Y[:,1], zeros(Float64,t),label ="sample")
    plot!(p_exact, -r, r, label ="exact")
    
    d=5
    R=1

    x=solve_opt(N,Y,t,R,d,delta=1,s=1,rho=1,numiter=1e4,eps=1e-4,tol_eig=1e-3,ball_cons=true);
    eval_PDF=func_eval_PDF(x,N,d,R,ball_cons=true)
    
    p_approx(z)=eval_PDF([z;zeros(N-1)])
    plot!(p_approx, -r, r, label = "d = $(d)",#=legend=:bottomright,=#title = "t = $(t)")
end