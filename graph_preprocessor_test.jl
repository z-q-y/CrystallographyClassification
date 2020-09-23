using Pkg

Pkg.activate("../../CrystalGraphConvNets.jl")

export inverse_square, exp_decay, build_graph, visualize_graph
include("../../CrystalGraphConvNets.jl/src/graph_functions.jl")

success_count = 0
total_count   = 0

try
    rm("graph_reader_test.txt")
end

io = open("graph_reader_test.txt", "w")

for (root, dirs, files) in walkdir("/home/qingyanz/cod/cif/")
    # cifs appear only in directories that are numbers
    filter!(dir->(tryparse(Int32, dir) == nothing)!=true, dirs)
    
    # If down to the lowest level
    if isempty(dirs)
        for file in files
            try
                @time build_graph(string(root, "/", file))
                success_count += 1
            catch e
            end
            
            total_count += 1
        end
        
        write(io, root)
        write(io, string("\n success_count: ", success_count))
        write(io, string("\n total_count: ", total_count, "\n"))
    end
end

close(io)

println("success_count: ", success_count)
println("total_count: ", total_count)