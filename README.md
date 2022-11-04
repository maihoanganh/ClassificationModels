# ClassificationModels
ClassificationModels is a Julia package of solving classification problem: 

Given a sequence of samples for each class ```K_j``` that is a subset of ```K```, determine which class ```K_j``` a given point in ```K``` belong to.

To tackle this problem, we utilize:
- Method based on Christoffel function.
- Method based on Maximum likelihood estimation. 


# Required softwares
ClassificationModels has been implemented on a desktop compute with the following softwares:
- Ubuntu 18.04.4
- Julia 1.3.1
- [Mosek 9.1](https://www.mosek.com)


# Installation
- To use ClassificationModels in Julia, run
```ruby
Pkg> add https://github.com/maihoanganh/ClassificationModels.git
```

# Usage
The following examples briefly guide to use ClassificationModels:

## Classification
Consider the following optimization polynomial problem:
```ruby

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

d=Vector{Int64}(undef,s) # degrees of polynomial estimations
R=Vector{Float64}(undef,s) # radius of ball containing the samples


using ClassificationModels

x=Vector{Vector{Float64}}(undef,s) # coefficients of polynomial estimations

for k=1:s
    println("Class ",k)
    println()
    d[k]=1
    R[k]=1
    
    # train a model
    x[k]=ClassificationModels.solve_opt(N,Y_train[k],t[k],R[k],d[k];
                                    delta=1,s=1,rho=1,numiter=1e2,
                                    eps=1e-2,tol_eig=1e-3,
                                    ball_cons=false,feas_start=false) # Maximum likelihood estimation
    println("------------")
end

eval_PDF=Vector{Function}(undef,s) # evaluate polynomial estimations

for k=1:s
    eval_PDF[k]=ClassificationModels.func_eval_PDF(x[k],N,d[k],R[k],ball_cons=false)
end

classifier(y)=findmax([eval_PDF[k](y) for k=1:s])[2]

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
    Lambda[k]=ClassificationModels.christoffel_func(N,Y_train[k],t[k],d[k],eps=0.0)
    println("------------")
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
```

See other examples from .ipynb files in the [link](https://github.com/maihoanganh/ClassificationModels/tree/main/test).


# References
For more details, please refer to:

**N. H. A. Mai, J.-B. Lasserre, V. Magron and S. Durasinovic. The Christoffel--Darboux and polynomially parametric classifiers for supervised learning. 2022. Forthcoming.**

To get the paper's benchmarks, download the zip file in this [link](https://drive.google.com/file/d/14yxm858LhCMkTCZopNlGDkqrUgMiJYwP/view?usp=sharing) and unzip the file.

The following codes are to run the paper's benchmarks:
```ruby

data="/home/hoanganh/Desktop/math-topics/algebraic_statistics/codes/datasets" # path of data 
#The path needs to be changed on the user's computer

using ClassificationModels

ClassificationModels.test_test()

ClassificationModels.test_Iris_MLE(data) # Table 1

ClassificationModels.univariate_Christoffel(data) # Figure 1
ClassificationModels.univariate_Christoffel2(data) # Figure 1

ClassificationModels.test_bivariate_Christoffel(data) # Figure 2
ClassificationModels.test_bivariate_Christoffel2(data) # Figure 2

ClassificationModels.univariate_MLE_density(data) # Figure 3
ClassificationModels.univariate_MLE_density2(data) # Figure 3

ClassificationModels.test_bivariate_MLE_density(data) # Figure 4
ClassificationModels.test_bivariate_MLE_density2(data) # Figure 4

ClassificationModels.univariate_MLE(data) # Figure 5
ClassificationModels.univariate_MLE2(data) # Figure 5

ClassificationModels.test_bivariate_MLE(data) # Figure 6
ClassificationModels.test_bivariate_MLE2(data) # Figure 6


ClassificationModels.test_Parkinson_Christoffel(data) # Section 5.1
ClassificationModels.test_Parkinson_Christoffel_arb_basis(data) # Section 5.1 (additional monomials)
ClassificationModels.test_Parkinson_MLE(data) # Section 5.1

ClassificationModels.test_optdigits_Christoffel(data) # Section 5.2
ClassificationModels.test_optdigits_MLE(data) # Section 5.2
ClassificationModels.test_optdigits_MLE_arb_basis(data) # Section 5.2 (additional monomials)
```
