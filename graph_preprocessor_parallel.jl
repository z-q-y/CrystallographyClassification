# Enable multiple cores
# Note that this goes before @everywhere functions
using Distributed
const NUMPROCS = Sys.CPU_THREADS - 1 # 13 if just assigned 14 cores
addprocs(NUMPROCS)
println("Number of available processes is ", nprocs())

@everywhere using Pkg
@everywhere Pkg.activate("/home/qingyanz/AtomicGraphNets.jl")
@everywhere using CSV
@everywhere using DataFrames
@everywhere using SimpleWeightedGraphs
@everywhere using DelimitedFiles
@everywhere using SharedArrays
@everywhere using ChemistryFeaturization
@everywhere using Serialization
@everywhere using Distributed
# @everywhere using ProgressBars
# @everywhere using PeriodicTable
# @everywhere import Dates

# Hyperparameters
datasize = "_all"
output_folder = "../newgraphs/$datasize"

# Acceptable labels
label_names = ["triclinic",
               "monoclinic",
               "orthorhombic",
               "tetragonal",
               "trigonal",
               "hexagonal",
               "cubic"]

#=
# If .cif contains these atoms, then graphs can't be built
max_atno = 83 # Do not consider post-Pb elements & noble gases 
nums_to_skip = union([2, 10, 18, 36, 54], Array(min(max_atno, 100):100))
skipped_els = [e.symbol for e in elements[nums_to_skip]]
=#

# Read the list of .cifs to be preprocessed and their labels
labels = CSV.File("../labels/example$datasize.txt", delim=", ") |> DataFrame

# Discard cifs with missing labels for time being
labels = labels[labels[:crystalsystem] .∈ Ref(label_names), :]

cif_roots = string.(labels[:, 1])

@everywhere function build_graphs_from_cif_list(cif_list::Vector{String}, output_folder::String)
    len_cif_list = length(cif_list)
    cifs_without_graphs = SharedArray{Int64}(len_cif_list)
    features  = Symbol.(["Group", "Row", "Block", "Atomic mass", "Atomic radius", "X"])
    nbins     = [18, 9, 4, 16, 10, 10]
    logspaced = [false, false, false, true, true, false]
    atom_featurevecs, featurization = make_feature_vectors(build_atom_feats(features; 
            nbins=nbins, logspaced=logspaced))
    
    # check if output folder exists, if not create it
    if !isdir(output_folder)
        mkdir(output_folder)
        @info "Output path provided did not exist, creating folder there."
    end

    @everywhere function cif_reader!(cif::String, idx::Int, 
            cifs_without_graphs, output_folder, atom_featurevecs, 
            featurization, len_cif_list; 
            use_voronoi=false, radius=8.0, max_num_nbr=12, 
            dist_decay_func=inverse_square, normalize=true)
        
        cif_dir = "../cif/"
        cif_path = string(cif_dir, cif[1], "/", cif[2:3], "/", 
                      cif[4:5], "/", cif, ".cif")

        local ag
        try 
            ag = build_graph(cif_path; use_voronoi=use_voronoi, 
            radius=radius, max_num_nbr=max_num_nbr, dist_decay_func=dist_decay_func, 
            normalize=normalize)
            @info("cif = $cif graph successfully built")
            
            add_features!(ag, atom_featurevecs, featurization)
            graph_path = joinpath(output_folder, string(cif, ".jls"))
            serialize(graph_path, ag)
        catch e
            cifs_without_graphs[idx] = parse(Int64, cif)
            progress = Float64(idx) / len_cif_list * 100
            @info("cif = $cif graph failed to build; progress = $progress %")
        end
    end
    
    pmap((a, b)->cif_reader!(a, b, cifs_without_graphs, output_folder, atom_featurevecs, featurization, len_cif_list),
        cif_roots, collect(1:len_cif_list), retry_delays=zeros(NUMPROCS))
    
    return cifs_without_graphs
end

cifs_without_graphs = build_graphs_from_cif_list(cif_roots, output_folder)

# Record cifs where graph building failed
writedlm("../newgraphs/cifs_without_graphs$datasize.csv", cifs_without_graphs)
