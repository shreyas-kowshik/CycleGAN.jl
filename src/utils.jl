function load_image(filename)
    img = load(filename)
    img = Float64.(channelview(img))
end

function load_dataset(path,imsize)
   imgs = []
   for r in readdir(path)
        img_path = string(path,r)
        push!(imgs,load_image(img_path))
    end
    reshape(hcat(imgs...),imsize,imsize,3,length(imgs))
end

function make_minibatch(X, idxs,size=256)
    """
    size : Image dimension
    """
    X_batch = Array{Float32}(undef, size, size, 3, length(idxs))
    for i in 1:length(idxs)
        X_batch[:,:,:, i] = Float32.(X[:,:,:,idxs[i]])
    end
    return X_batch
end

function nullify_grad!(p)
  if typeof(p) <: TrackedArray
    p.grad .= 0.0f0
  end
  return p
end

function zero_grad!(model)
  model = mapleaves(nullify_grad!, model)
end

function norm(x)
    2.0 .* x .- 1.0
end

function denorm(x)
   (x .+ 1.0)./(2.0) 
end

expand_dims(x,n::Int) = reshape(x,ones(Int64,n)...,size(x)...)
function squeeze(x) 
    if size(x)[end] != 1
        return dropdims(x, dims = tuple(findall(size(x) .== 1)...))
    else
        # BATCH_SIZE = 1
        int_val = dropdims(x, dims = tuple(findall(size(x) .== 1)...))
        return reshape(int_val,size(int_val)...,1)
    end
end

drop_first_two(x) = dropdims(x,dims=(1,2))

# BatchNorm Wrapper
function BatchNormWrap(out_ch)
    Chain(x->expand_dims(x,2),
    BatchNorm(out_ch),
    x->squeeze(x))
end