using Test
using Images,CuArrays,Flux
using Flux:@treelike, Tracker
using Base.Iterators: partition
using Random
using Statistics

include("../src/discriminator.jl")
include("../src/generator.jl")
include("../src/utils.jl")

@test begin
    g = UNet() |> gpu
    a = ones(256,256,3,2) |> gpu
    out = g(a)
    size(out) == (256,256,3,2)
end

@test begin
    d = Discriminator() |> gpu
    a = ones(256,256,3,2) |> gpu
    out = drop_first_two(d(a))
    size(out) == (1,2)
end