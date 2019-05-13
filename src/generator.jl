UNetConvBlock(in_chs, out_chs, kernel = (3, 3)) =
    Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1)) ,BatchNormWrap(out_chs)...,x->leakyrelu.(x),
          Conv(kernel, out_chs=>out_chs,pad = (1, 1)),BatchNormWrap(out_chs)...,x->leakyrelu.(x))

struct UNetUpBlock
    upsample
    conv_layer
end

@treelike UNetUpBlock

UNetUpBlock(in_chs::Int, out_chs::Int, kernel = (3, 3)) =
    UNetUpBlock(ConvTranspose((2, 2), in_chs=>out_chs, stride=(2, 2)),
                Chain(Conv(kernel, in_chs=>out_chs,pad = (1, 1)),BatchNormWrap(out_chs)...,x->leakyrelu.(x),
                Conv(kernel, out_chs=>out_chs,pad = (1, 1)),BatchNormWrap(out_chs)...,x->leakyrelu.(x)))

function (u::UNetUpBlock)(x, bridge)
    x = u.upsample(x)
    u.conv_layer(cat(x, bridge, dims = 3))
end

struct UNet
    pool_layer
    conv_blocks
    up_blocks
end

@treelike UNet

# This is to be used for Background and Foreground segmentation
function UNet()
    pool_layer = MaxPool((2, 2))
    conv_blocks = (UNetConvBlock(3, 64), UNetConvBlock(64, 128), UNetConvBlock(128, 256),
                   UNetConvBlock(256, 512), UNetConvBlock(512, 1024))
    up_blocks = (UNetUpBlock(1024, 512), UNetUpBlock(512, 256), UNetUpBlock(256, 128),
                 UNetUpBlock(128, 64), Conv((1, 1), 64=>3))
    UNet(pool_layer, conv_blocks, up_blocks)
end

function (u::UNet)(x)
    outputs = Vector(undef, 5)
    outputs[1] = u.conv_blocks[1](x)
    for i in 2:5
        pool_x = u.pool_layer(outputs[i - 1])
        outputs[i] = u.conv_blocks[i](pool_x)
    end
    up_x = outputs[end]
    for i in 1:4
        up_x = u.up_blocks[i](up_x, outputs[end - i])
    end
    tanh.(u.up_blocks[end](up_x))
end