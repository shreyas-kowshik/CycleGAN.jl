using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics

include("utils.jl")
include("generator.jl")
include("discriminator.jl")

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 1
dis_lr = 0.0001f0
gen_lr = 0.0001f0
λ₁ = 10.0 # Cycle loss weight for dommain A
λ₂ = 10.0 # Cycle loss weight for domain B
NUM_EXAMPLES = 2 # Temporary for experimentation
VERBOSE_FREQUENCY = 2 # Verbose output after every 2 epochs

# Data Loading
dataA = load_dataset("../data/trainA/",256)[:,:,:,1:NUM_EXAMPLES] |> gpu
dataB = load_dataset("../data/trainB/",256)[:,:,:,1:NUM_EXAMPLES] |> gpu
mb_idxs = partition(shuffle!(collect(1:size(dataA)[end])), BATCH_SIZE)
train_A = [make_minibatch(dataA, i) for i in mb_idxs]
train_B = [make_minibatch(dataB, i) for i in mb_idxs]
println("Loaded Data")

# Define Optimizers
opt_gen = ADAM(gen_lr,(0.5,0.999))
opt_disc_A = ADAM(dis_lr,(0.5,0.999))
opt_disc_B = ADAM(dis_lr,(0.5,0.999))

# Define models
gen_A = UNet() |> gpu # Generator For A->B
gen_B = UNet() |> gpu # Generator For B->A
dis_A = Discriminator() |> gpu # Discriminator For Domain A
dis_B = Discriminator() |> gpu # Discriminator For Domain B
println("Loaded Models")


# Forward prop, backprop, optimise!
function train_step(X_A,X_B) 
    # LABELS #
    real_labels = ones(1,BATCH_SIZE)
    fake_labels = ones(0,BATCH_SIZE)
    
    ### Forward Propagation ###
    # zero_grad!(gen_A)
    # zero_grad!(gen_B)
    
    println("1")
    fake_B = gen_A(X_A) # Fake image generated in domain B
    fake_B_prob = drop_first_two(dis_B(fake_B)) # Probability that generated image in domain B is real
    real_B_prob = drop_first_two(dis_B(X_B)) # Probability that original image in domain B is real
    println("2")
    
    fake_A = gen_B(X_B) # Fake image generated in domain A
    fake_A_prob = drop_first_two(dis_A(fake_A)) # Probability that generated image in domain A is real
    real_A_prob = drop_first_two(dis_A(X_A)) # Probability that original image in domain A is real
    
    println("3")
    rec_A = gen_B(fake_B)
    rec_B = gen_A(fake_A)
    
    ### Generator Losses ###
    # For domain A->B  #
    println("4")
    gen_B_loss = mean((fake_B_prob .- real_labels).^2)
    rec_B_loss = mean(abs.(X_B .- rec_B)) # Reconstruction loss for domain B
    
    # For domain B->A  #
    println("5")
    gen_A_loss = mean((fake_A_prob .- real_labels).^2)
    rec_A_loss = mean(abs.(X_A .- rec_A)) # Reconstrucion loss for domain A
    
    gen_loss = gen_A_loss + gen_B_loss + λ₁*rec_A_loss + λ₂*rec_B_loss # Total generator loss
    
    println("Forward propagate generators")
    # Optimise
    gs = Tracker.gradient(() -> gen_loss,params(params(gen_A)...,params(gen_B)...))
    update!(opt_gen,params(params(gen_A)...,params(gen_B)...),gs)
    println("Optimised generators")
    
    ### Discriminator Losses ###
    # For domain A #
    # zero_grad!(dis_A)
    fake_A_prob = drop_first_two(dis_A(fake_A.data))
    dis_A_real_loss = mean((real_A_prob .- real_labels).^2)
    dis_A_fake_loss = mean((fake_A_prob .- fake_labels).^2)
    dis_A_loss = 0.5 * (dis_A_real_loss + dis_A_fake_loss)
    println("Forward propagate disA")
    gs = Tracker.gradient(() -> dis_A_loss,params(dis_A))
    update!(opt_disc_A,params(dis_A),gs)
    println("Optimised disA")
    
    # For domain B #
    # zero_grad!(dis_B)
    fake_B_prob = drop_first_two(dis_B(fake_B.data))
    dis_B_real_loss = mean((real_B_prob .- real_labels).^2)
    dis_B_fake_loss = mean((fake_B_prob .- fake_labels).^2)
    dis_B_loss = 0.5 * (dis_B_real_loss + dis_B_fake_loss)
    println("Forward propagate disB")
    gs = Tracker.gradient(() -> dis_B_loss,params(dis_B))
    update!(opt_disc_B,params(dis_B),gs)
    println("Optimised disB")
    
    return gen_loss,dis_A_loss,dis_B_loss
end

println("Training...")
for epoch in 1:NUM_EPOCHS
    println("-----------Epoch : $epoch-----------")
    for i in 1:length(train_A)
        g_loss,dA_loss,dB_loss = train_step(train_A[i],train_B[i])
        if epoch % VERBOSE_FREQUENCY == 0
            println("Gen Loss : $g_loss")
            println("DisA Loss : $dA_loss")
            println("DisB Loss : $dB_loss")
        end
    end
end