using Pkg
Pkg.activate("/home/qingyanz/CrystalGraphConvNets.jl")

using CSV
using DataFrames
using SparseArrays
using Random, Statistics
using Flux
using Flux: @epochs
using Flux: kldivergence
using Flux: logitcrossentropy
using Flux: crossentropy
using GeometricFlux
using SimpleWeightedGraphs
using CrystalGraphConvNets
using DelimitedFiles
using Distributed
import Dates

using PyCall

s = pyimport("pymatgen.core.structure")

import Flux:@functor

import CrystalGraphConvNets.inverse_square
include("../../CrystalGraphConvNets.jl/src/graph_functions.jl")

# How many .cifs should there be in the dataset?
#     _small: ~1000
#     _mid:   ~10000
#     _large: ~20000
#     _all:   ~400000
datasize = "_mid";

# Hyperparameters
const train_frac = 0.8
const num_epochs = 8
const cutoff_radius = 8.0
const max_num_nbr = 12
decay_fcn=inverse_square

#=
prop = "crystalsystem"
cif_dir = "../cif/"
graph_dir = "../graphs/"
=#

num_conv = 3
atom_fea_len = 32
pool_type = "mean"
crys_fea_len = 128
num_hidden_layers = 1
#lrs = exp10.(range(-3.2,stop=-2.8,length=5))
lr = 0.001

features = ["group", "row", "X", "atomic_radius", "block"]
num_bins = [18, 8, 10, 10, 4]
logspaced = [false, false, false, true, false]
num_features = sum(num_bins)

cifs_without_graphs = readdlm("../graphs/cifs_without_graphs$datasize.csv", ',', Int32)
labels = CSV.read("../labels/example$datasize.txt", type="String", delim=", ")

# Remove cifs with corrupted or out-of-scope labels
labels = labels[labels[:crystalsystem] .∉ Ref(["?", "i", "!"]), :];

# remove cifs without feasible graphs
labels = labels[setdiff(1:end, cifs_without_graphs[:, 1]), :];
        
atom_feature_vecs = make_feature_vectors(features, num_bins, logspaced)
element_lists = Array{String}[]
inputs = FeaturedGraph{SimpleWeightedGraph{Int64, Float32}, Array{Float32,2}}[]
cifs_beyond_scope = Matrix(undef, 0, 2)

function graph_reader!(inputs, element_lists, cifs_beyond_scope, atom_feature_vecs, datasize, cif_roots)
    prop = "crystalsystem"
    cif_dir = "../cif/"
    graph_dir = "../graphs/"
    
    for (row, cif) in enumerate(cif_roots)
        # path to the cif
        gr_path = string(graph_dir, "grwts", datasize, "/", cif, ".txt")
        el_path = string(graph_dir, "ellists", datasize, "/", cif, ".txt")

        try
            gr = SimpleWeightedGraph(Float32.(readdlm(gr_path)))
                    # restore to Float32
            els = readdlm(el_path)

            feature_mat = hcat([atom_feature_vecs[e] for e in els]...)
            input = FeaturedGraph(gr, feature_mat)
            push!(inputs, input)
            push!(element_lists, els)
        catch e
            println("unable to build graph at cif = ", cif)

            cifs_beyond_scope = vcat(cifs_beyond_scope, [row cif])
        end
    end
    
    println("Graph reading completed for ", length(cif_roots), " graphs")
    
    return inputs, element_lists, cifs_beyond_scope
end

inputs, element_lists, cifs_beyond_scope = graph_reader!(inputs, element_lists, 
                                                         cifs_beyond_scope, 
                                                         atom_feature_vecs,
                                                         datasize,
                                                         labels[:, 1])

labels = labels[setdiff(1:end, cifs_beyond_scope[:, 1]), :]
cif_roots = labels[:, 1];
y = labels[:, 2]

# Redefine num_pts and some other hyperparameters here
num_pts = size(labels)[1] # for now
num_train = Int32(round(train_frac * num_pts))
num_test = num_pts - num_train

# Sample from y
indices = shuffle(1:size(labels,1))[1:num_pts]
ysample = y[indices]
input   = inputs[indices]
cif_roots_sample = cif_roots[indices]

ycat = collect(Set(ysample)) # Categorization of y
ycat_len = length(ycat)
output = Flux.onehotbatch(ysample, ycat)
# output = Flux.onehotbatch(ycat, ysample)

train_output = output[:, 1:num_train];
test_output  = output[:, num_train+1:end];
train_input  = input[1:num_train];
test_input   = input[num_train+1:end];
train_y      = ysample[1:num_train];
test_y       = ysample[num_train+1:end];
train_data   = zip(train_input, train_output);

train_input = reshape(train_input, 1, length(train_input))
test_input  = reshape(test_input, 1, length(test_input))
        
# train_data   = zip(train_input, train_output);

#=
train_output = output[1:num_train, :];
test_output = output[num_train+1:end, :];
train_input = input[1:num_train];
test_input = input[num_train+1:end];
train_data = zip(train_input, train_output);
=#

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

# Define the model
# Average pool
model = Chain(CGCNConv(num_features=>atom_fea_len), 
              [CGCNConv(atom_fea_len=>atom_fea_len) for i in 1:num_conv-1]..., 
              CGCNMeanPool(crys_fea_len, 0.1), 
              [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers-1]...,
              Dense(crys_fea_len, ycat_len), softmax);

#= 
# Max pool
model = Chain([CGCNConv(num_features=>num_features) for i in 1:num_conv]...,
              CGCNMaxPool(crys_fea_len, 0.1),
              [Dense(crys_fea_len, crys_fea_len, softplus) for i in 1:num_hidden_layers-1]..., 
              Dense(crys_fea_len, ycat_len), softmax);
=#

# Kullback-Leiber & Cross Entropy
loss_kl(x, y) = Flux.kldivergence(model(x), [y])
loss_ce(x, y) = Flux.logitcrossentropy(model(x), [y])
# loss_mse(x, y) = Flux.mse(model(x), y)
evalcb() = @show(mean(loss_kl.(test_input, test_output)))
start_err = evalcb()

println("Training started with $num_pts data points")
Dates.now()

# train
opt = ADAM(lr)
_, train_time, mem, _, _ = @timed @epochs num_epochs Flux.train!(loss_kl, 
            params(model), train_data, opt, cb=Flux.throttle(evalcb, 5))

end_err = evalcb()
try
    end_ce = @show(mean(loss_ce.(test_input, test_output)))
catch e
    @info("loss_ce() not working")
end
 
# write results
try
    writedlm("../out/raw_predictions_datasize_$(num_pts)_train.csv", 
                model.(train_input))
    writedlm("../out/raw_predictions_datasize_$(num_pts)_test.csv", 
                model.(test_input))
catch e
    @info("Unable to write raw results")
end

ycat_dict = Dict(zip(1:ycat_len, ycat))
test_pred = model.(test_input)
test_pred_indices = mapslices(argmax, test_pred, dims=1)

correct_count = 0
prediction_labels = Matrix(undef, 0, 3)

function prediction_stats!(pred_indices, pred_dict, y, cif_roots,
                           correct_count, prediction_labels)
    for (i, index) in enumerate(pred_indices)
        pred = pred_dict[index]
        label = y[i]
        cif_name = cif_roots[num_train+1:end][i]

        if pred == label
            correct_count += 1
        end

        prediction_labels = vcat(prediction_labels, [cif_name pred label])
    end
    
    return correct_count, prediction_labels
end

correct_count, prediction_labels = prediction_stats!(test_pred_indices,
                                                     ycat_dict, 
                                                     test_y, cif_roots_sample,
                                                     correct_count, 
                                                     prediction_labels)

writedlm("../out/predictions_datasize_$(num_pts)_test.csv", prediction_labels)

accuracy = correct_count / num_test
println("The accuracy of our prediction is $accuracy")

#------------------------------80-characters-----------------------------------#