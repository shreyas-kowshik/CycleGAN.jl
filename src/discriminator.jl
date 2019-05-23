# weight initialization
function random_normal(shape...)
    return map(Float32,rand(Normal(0,0.02),shape...))
end

ConvBlock(in_ch::Int,out_ch::Int) = 
    Chain(Conv((6,6), in_ch=>out_ch,pad = (2, 2), stride=(2,2);init=random_normal),
          BatchNormWrap(out_ch)...,
          x->leakyrelu.(x,0.2))

function Discriminator()
    model = Chain(Conv((6,6), 6=>64,pad = (2, 2), stride=(2,2);init=random_normal),BatchNormWrap(64)...,x->leakyrelu.(x,0.2),
                  ConvBlock(64,128),
                  ConvBlock(128,256),
                  ConvBlock(256,512),
                  ConvBlock(512,256),
                  ConvBlock(256,128),
                  ConvBlock(128,64),
                  Conv((4,4), 64=>1,pad = (1, 1), stride=(2,2);init=random_normal),
                  x->σ.(x))
    return model 
end