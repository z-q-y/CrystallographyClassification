
using Pkg
Pkg.activate("/home/qingyanz/AtomicGraphNets.jl")

using CSV
using DataFrames
using SparseArrays
using Random, Statistics
using Flux
using Flux: @epochs
using CuArrays
using SimpleWeightedGraphs
using AtomicGraphNets
using DelimitedFiles
using Distributed
using ProgressBars
using ChemistryFeaturization
using Serialization
import Dates

using PyCall

# s = pyimport("pymatgen.core.structure")

import Flux:@functor

# import CrystalGraphConvNets.inverse_square
# include("../../CrystalGraphConvNets.jl/src/graph_functions.jl")

#= Before defining the model...
  All floats in the CrystalGraphConvNets.jl package are defined as Float32. 
  However certain loss functions given by Flux yield Float64 gradients when
  given Float32 inputs. This prevents subsequent steps from taking advantage
  of OpenBLAS shortcuts at best, and causes type errors at worst.
  
  The following line converts all floats in the NN model to Float32. Dhairya
  claims this is a drastic but necessary stop gap measure. See more at
      https://github.com/FluxML/Flux.jl/issues/514
=#
Base.promote_rule(::Type{Float64}, ::Type{Float32}) = Float32

# How many .cifs should there be in the dataset?
#     _mini: ~1000
#     _small: ~3500
#     _mid:   ~9000
#     _large: ~20000
#     _long:  ~50000
#     _all:   ~400000
datasize = "_mid";

# Hyperparameters
const train_frac = 0.8
const num_epochs = 8
const cutoff_radius = 8.0
const max_num_nbr = 12
decay_fcn=inverse_square

prop = "crystalsystem"
cif_dir = "../cif/"
graph_dir = "../newgraphs/$datasize/"

num_conv = 3
atom_fea_len = 32
pool_type = "mean"
crys_fea_len = 128
num_hidden_layers = 1
# lrs = exp10.(range(-3.2,stop=-2.8,length=5))
lr = 0.005 # 0.001

# features taken from ChemistryFeaturization.jl/data/pymatgen_atom_data.csv
features = ["Group", "Row", "X", "Atomic radius", "Block"]
num_bins = [18, 8, 10, 10, 4]
logspaced = [false, false, false, true, false]

cifs_without_graphs = readdlm("../graphs/cifs_without_graphs$datasize.csv", ',', Int32);
labels = CSV.File("../labels/example$datasize.txt", delim=", ") |> DataFrame;

# Remove cifs with corrupted or out-of-scope labels
label_names = ["triclinic",
               "monoclinic",
               "orthorhombic",
               "tetragonal",
               "trigonal",
               "hexagonal",
               "cubic"];

labels = labels[labels[:crystalsystem] .∈ Ref(label_names), :];

# remove cifs without feasible graphs
labels = labels[setdiff(1:end, cifs_without_graphs[:, 1]), :];
cif_roots = string.(labels[:, 1]);
y = string.(labels[:, 2])

inputs = AtomGraph[]
out_of_scope_cifs = []

for (row, cif) in enumerate(cif_roots)
    try
        input = deserialize(string(graph_dir, cif, ".jls"))
        push!(inputs, input)
    catch e
        push!(out_of_scope_cifs, [row cif])
        @info("graph building failed at cif = $cif")
    end
    
    if iszero(row % 100)
        @info("milestone: $row cifs processed ")
    end
end

out_of_scope_cifs = vcat(out_of_scope_cifs...)
labels = labels[setdiff(1:end, out_of_scope_cifs[:, 1]), :];
num_features = size(inputs[1].features)[1]


num_pts = size(labels)[1] # 500
num_train = Int32(round(train_frac * num_pts))
num_test = num_pts - num_train

# Sample from y
indices = shuffle(1:size(labels,1))[1:num_pts]
ysample = y[indices]
input   = inputs[indices]
cif_roots_sample = cif_roots[indices];

ycat = collect(Set(ysample)) # Categorization of y
ycat_len = length(ycat)
output = Flux.onehotbatch(ysample, ycat)
output = [output[:, i] for i in 1:size(output, 2)]

train_output = output[1:num_train, :] |> gpu;
test_output  = output[num_train+1:end, :] |> gpu;
train_input  = input[1:num_train] |> gpu;
test_input   = input[num_train+1:end] |> gpu;
train_y      = ysample[1:num_train];
test_y       = ysample[num_train+1:end];
train_cifs   = cif_roots_sample[1:num_train];
test_cifs    = cif_roots_sample[num_train+1:end];
train_data   = zip(train_input, train_output);

# Average pool
model = Chain(AGNConv(num_features=>atom_fea_len), 
              [AGNConv(atom_fea_len=>atom_fea_len) for i in 1:num_conv-1]..., 
              AGNMeanPool(crys_fea_len, 0.1), 
              [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers-1]...,
              Dense(crys_fea_len, ycat_len), softmax) |> gpu;

#= 
# Max pool
model = Chain([CGCNConv(num_features=>num_features) for i in 1:num_conv]...,
              CGCNMaxPool(crys_fea_len, 0.1),
              [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers-1]..., 
              Dense(crys_fea_len, ycat_len), softmax);
=#

# Kullback-Leiber & Cross Entropy
loss_kl(x, y) = Flux.kldivergence(model(x), y)
loss_lce(x, y) = Flux.logitcrossentropy(model(x), y)
evalcb() = @show(mean(loss_lce.(test_input, test_output)))
start_err = evalcb()

println("Training started with $num_pts data points")
Dates.now()

# train
opt = ADAM(lr)
_, train_time, mem, _, _ = @timed @epochs num_epochs Flux.train!(loss_lce, params(model), train_data, opt, cb=Flux.throttle(evalcb, 5))

end_err = evalcb()
end_ce = @show(mean(loss_lce.(test_input, test_output)))

# write results
try
    writedlm("../out/raw_predictions_datasize_$(num_pts)_train.csv", model.(train_input))
    writedlm("../out/raw_predictions_datasize_$(num_pts)_test.csv", model.(test_input))
catch e
    @info("Unable to write raw results")
end

labeldict_num2txt = Dict(zip(1:ycat_len, ycat))
labeldict_txt2num = Dict(zip(ycat, 1:ycat_len))
test_pred         = model.(test_input)
test_pred_indices = getindex.(argmax.(test_pred), 1)

correct_count = sum([labeldict_txt2num[label] for label in test_y] .== test_pred_indices)
accuracy = correct_count / num_test

prediction_labels = zip(test_cifs,
    [labeldict_num2txt[idx] for idx in test_pred_indices],
    test_y)
prediction_labels = permutedims(hcat(map(x->collect(x), prediction_labels)...))

writedlm("../out/predictions_datasize_$(num_pts)_test.csv", prediction_labels)

println("The accuracy of our prediction is $accuracy")