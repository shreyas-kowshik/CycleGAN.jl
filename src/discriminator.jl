ConvBlock(in_ch::Int,out_ch::Int) = 
    Chain(Conv((4,4), in_ch=>out_ch,pad = (1, 1), stride=(2,2)),
          BatchNormWrap(out_ch)...,
          x->leakyrelu.(x))

function Discriminator()
    model = Chain(Conv((4,4), 3=>8,pad = (1, 1), stride=(2,2)),x->leakyrelu.(x)) 
    model = Chain(model...,ConvBlock(8,16)...)
    model = Chain(model...,ConvBlock(16,32)...)
    model = Chain(model...,ConvBlock(32,64)...)
    model = Chain(model...,ConvBlock(64,32)...)
    model = Chain(model...,ConvBlock(32,16)...)
    model = Chain(model...,ConvBlock(16,8)...)
    model = Chain(model...,Chain(Conv((4,4), 8=>1,pad = (1, 1), stride=(2,2))))
    # Note : No sigmoid on last layer of Discriminator
    #        LSGAN can handle it with basic backpropagation
    model
end