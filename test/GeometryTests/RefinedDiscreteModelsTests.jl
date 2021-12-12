module RefinedDiscreteModelsTests

using Test
using Gridap.Arrays
using Gridap.Fields
using Gridap.ReferenceFEs
using Gridap.Geometry
using Gridap.Geometry: DiscreteModelMock
using Gridap.Visualization


function random_estimator(ncells)
    rand(ncells)
end

function constant_estimator(c, ncells)
    fill(c, ncells)
end


domain = (0,1,0,1)
partition = (1,1)
model = CartesianDiscreteModel(domain,partition)
model = simplexify(model)
cell_map = get_cell_map(get_triangulation(model))
ncells = length(cell_map)
η_arr = random_estimator(ncells)
Nsteps = 15
model_refs = Vector{DiscreteModel}(undef, Nsteps)
sort_flag = true
θ = 0.5
model_refs[1] = newest_vertex_bisection(model, η_arr; sort_flag=sort_flag, θ=θ)
sort_flag = false
for i in 1:Nsteps-1
    cell_map = get_cell_map(get_triangulation(model_refs[i]))
    ncells = length(cell_map)
    #η_arr = rand(ncells)
    η_arr = random_estimator(ncells)
    model_refs[i + 1] = newest_vertex_bisection(model_refs[i], η_arr, sort_flag=sort_flag, θ=θ)
end
writevtk(Triangulation(model_refs[end]), "refined")

@test model_refs[end] isa DiscreteModel
end # module
