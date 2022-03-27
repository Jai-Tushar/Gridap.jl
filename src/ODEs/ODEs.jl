"""

The exported names are
$(EXPORTS)
"""
module ODEs

using DocStringExtensions

include("ODETools/ODETools.jl")

include("TransientFETools/TransientFETools.jl")

# include("DiffEqsWrappers/DiffEqsWrappers.jl")

end #module

const GridapODEs = ODEs
