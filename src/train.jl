using Images,CuArrays,Flux
using Flux:@treelike, Tracker, update!
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
λ₁ = convert(Float32,10.0) # Cycle loss weight for dommain A
λ₂ = convert(Float32,10.0) # Cycle loss weight for domain B
λid = convert(Float32,1.0) # Identity loss weight - Set this to '0' if identity loss is not required
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

function dA_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    # LABELS #
    real_labels = ones(1,BATCH_SIZE) |> gpu
    fake_labels = zeros(1,BATCH_SIZE) |> gpu

    fake_A = gen_B(b) # Fake image generated in domain A
    fake_A_prob = drop_first_two(dis_B(fake_A.data)) # Probability that generated image in domain A is real
    real_A_prob = drop_first_two(dis_B(a)) # Probability that original image in domain A is real

    dis_A_real_loss = mean((real_A_prob .- real_labels).^2)
    dis_A_fake_loss = mean((fake_A_prob .- fake_labels).^2)
    0.5 * (dis_A_real_loss + dis_A_fake_loss)
end

function dB_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    # LABELS #
    real_labels = ones(1,BATCH_SIZE) |> gpu
    fake_labels = zeros(1,BATCH_SIZE) |> gpu

    fake_B = gen_A(a) # Fake image generated in domain B
    fake_B_prob = drop_first_two(dis_B(fake_B.data)) # Probability that generated image in domain B is real
    real_B_prob = drop_first_two(dis_B(b)) # Probability that original image in domain B is real

    dis_B_real_loss = mean((real_B_prob .- real_labels).^2)
    dis_B_fake_loss = mean((fake_B_prob .- fake_labels).^2)
    0.5 * (dis_B_real_loss + dis_B_fake_loss)
end

function g_loss(a,b)
    """
    a : Image in domain A
    b : Image in domain B
    """
    # LABELS #
    real_labels = ones(1,BATCH_SIZE) |> gpu
    fake_labels = zeros(1,BATCH_SIZE) |> gpu

    # Forward Propogation # 
    fake_B = gen_A(a) # Fake image generated in domain B
    fake_B_prob = drop_first_two(dis_B(fake_B)) # Probability that generated image in domain B is real
    real_B_prob = drop_first_two(dis_B(b)) # Probability that original image in domain B is real

    fake_A = gen_B(b) # Fake image generated in domain A
    fake_A_prob = drop_first_two(dis_A(fake_A)) # Probability that generated image in domain A is real
    real_A_prob = drop_first_two(dis_A(a)) # Probability that original image in domain A is real
    
    rec_A = gen_B(fake_B)
    rec_B = gen_A(fake_A)
    
    ### Generator Losses ###
    # For domain A->B  #
    gen_B_loss = mean((fake_B_prob .- real_labels).^2)
    rec_B_loss = mean(abs.(b .- rec_B)) # Reconstruction loss for domain B
    
    # For domain B->A  #
    gen_A_loss = mean((fake_A_prob .- real_labels).^2)
    rec_A_loss = mean(abs.(a .- rec_A)) # Reconstrucion loss for domain A

    # Identity losses 
    # gen_A should be identity if b is fed : ||gen_A(b) - b||
    idt_A_loss = mean(abs.(gen_A(b) .- b))
    # gen_B should be identity if a is fed : ||gen_B(a) - a||
    idt_B_loss = mean(abs.(gen_B(a) .- a))

    gen_A_loss + gen_B_loss + λ₁*rec_A_loss + λ₂*rec_B_loss  + λid*(λ₁*idt_A_loss + λ₂*idt_B_loss)
end

# Forward prop, backprop, optimise!
function train_step(X_A,X_B) 
    # Normalise the Images
    X_A = norm(X_A)
    X_B = norm(X_B)

    # Optimise Discriminators
    gs = Tracker.gradient(() -> dA_loss(X_A,X_B),params(dis_A))
    update!(opt_disc_A,params(dis_A),gs)

    gs = Tracker.gradient(() -> dB_loss(X_A,X_B),params(dis_B))
    update!(opt_disc_B,params(dis_B),gs)

    # Optimise Generators
    gs = Tracker.gradient(() -> g_loss(X_A,X_B),params(params(gen_A)...,params(gen_B)...))
    update!(opt_gen,params(params(gen_A)...,params(gen_B)...),gs)

    # Forward propagate to collect the losses
    gloss = g_loss(X_A,X_B)
    dAloss = dA_loss(X_A,X_B)
    dBloss = dB_loss(X_A,X_B)

    return gloss,dAloss,dBloss
end

function train()
    println("Training...")
    for epoch in 1:NUM_EPOCHS
        println("-----------Epoch : $epoch-----------")
        for i in 1:length(train_A)
            g_loss,dA_loss,dB_loss = train_step(train_A[i] |> gpu,train_B[i] |> gpu)
            if epoch % VERBOSE_FREQUENCY == 0
                println("Gen Loss : $g_loss")
                println("DisA Loss : $dA_loss")
                println("DisB Loss : $dB_loss")
            end
        end
    end
end

### SAMPLING ###
function sampleA2B(X_A_test)
    """
    Samples new images in domain B
    X_A_test : N x C x H x W array - Test images in domain A
    """
    testmode!(gen_A)
    X_A_test = norm(X_A_test)
    X_B_generated = cpu(denorm(gen_A(X_A_test |> gpu)).data)
    testmode!(gen_A,false)
    imgs = []
    s = size(X_B_generated)
    for i in size(X_B_generated)[end]
       push!(imgs,colorview(RGB,reshape(X_B_generated[:,:,:,i],3,s[1],s[2])))
    end
    imgs
end

function test()
   # load test data
   dataA = load_dataset("../data/trainA/",256)[:,:,:,1:2] |> gpu
   out = sampleA2B(dataA)
   for (i,img) in enumerate(out)
        save("../sample/A_$i.png",img)
   end
end

train()