module CUPTI

using CUDAdrv: CUcontext, CUstream, CUdevice, CUdevice_attribute,
               CUgraph, CUgraphNode, CUgraphNodeType, CUgraphExec

using ..CUDAnative: libcupti

# TODO: move to CUDAdrv
struct CUuuid
    bytes::NTuple{16,Int8}
end

using CEnum

include("libcupti_common.jl")
include("error.jl")

include("libcupti.jl")

end
