using Pkg
Pkg.activate("/home/qingyanz/CrystalGraphConvNets.jl")

using CSV
using DataFrames
using SimpleWeightedGraphs
using CrystalGraphConvNets
using DelimitedFiles
using ProgressBars
using PeriodicTable
using Distributed
import Dates

using PyCall
s = pyimport("pymatgen.core.structure")

# export inverse_square, exp_decay, build_graph, visualize_graph
include("../../CrystalGraphConvNets.jl/src/graph_functions.jl")

# Hyperparameters
const NUMPROCS = 13
const time_limit = 200 - 1 # time limit for slurm job in hours

train_frac = 0.8
num_epochs = 10
cutoff_radius = 8.0
max_num_nbr=12
decay_fcn=inverse_square

datasize = "_alll"
cifs_without_graphs = Matrix(undef, 0, 2)

# If .cif contains these atoms, then graphs can't be built
max_atno = 83 # Do not consider post-Pb elements & noble gases 
nums_to_skip = union([2, 10, 18, 36, 54], Array(min(max_atno, 100):100))
skipped_els = [e.symbol for e in elements[nums_to_skip]]

# Read the list of .cifs to be preprocessed and their labels
labels = CSV.read("../labels/example$datasize.txt",
                  type="String", delim=", ")

# Discard cifs with missing labels for time being
labels = labels[labels[:crystalsystem] .∉ Ref(["?", "i", "!"]), :]

# Enable multiple cores
# Note that this goes before @everywhere functions
addprocs(NUMPROCS)
println("Number of available processes is ", nprocs())

@everywhere function cif_reader!(cif_root, cifs_without_graphs, datasize, skipped_els)
    tic = Dates.now()
    toc = Dates.now()
    
    # make all the combinations
    # params = Iterators.product(cutoff_radii, nbr_num_cutoffs, decay_fcns)        
    # for p in params
    
    params_suffix = ""
    cif_dir = "../cif/"
    el_list_dir = "../graphs/ellists$datasize"
    graph_weights_dir = "../graphs/grwts$datasize"

    for (row, cif) in ProgressBar(enumerate(cif_root))
        # path to the cif
        cif_path = string(cif_dir, cif[1], "/", cif[2:3], "/", 
                          cif[4:5], "/", cif, ".cif")

        try
            el_list_subdir = string(el_list_dir, params_suffix,'/')
            gr_wt_subdir = string(graph_weights_dir, params_suffix,'/')

            if !isdir(el_list_subdir)
                mkdir(el_list_subdir)
            end
            if !isdir(gr_wt_subdir)
                mkdir(gr_wt_subdir)
            end

            gr, els = @time build_graph(cif_path) 
                      # radius=radius, max_num_nbr=max_num_nbr,
                      # dist_decay_func=decay_func

            out_of_scope_els = intersect(Set(els), skipped_els)

            if isempty(out_of_scope_els)
                writedlm(string(gr_wt_subdir, cif,".txt"), gr.weights)
                writedlm(string(el_list_subdir, cif,".txt"), els)
            else
                @info("cif = $cif contains out of scope element $(out_of_scope_els)")
                cifs_without_graphs = vcat(cifs_without_graphs, [row cif])
            end
        catch e
            @info("unable to build graph at cif = $cif")

            # Record cifs where graph building failed
            cifs_without_graphs = vcat(cifs_without_graphs, [row cif])
        end

        toc = Dates.now()
        if (toc - tic) > Dates.Hour(time_limit)
            break
        end
    end
    
    return cifs_without_graphs
end

# cifs_without_graphs = pmap(cif_reader!, (labels[:, 1], cifs_without_graphs,
#                                          datasize, skipped_els, time_limit))

cifs_without_graphs = pmap((a, cifs_without_graphs, c,d)->cif_reader!(a,b,c,d), labels[:, 1],
                            cifs_without_graphs, datasize, skipped_els)

# Record cifs where graph building failed
writedlm("../graphs/cifs_without_graphs$datasize.csv", cifs_without_graphs)