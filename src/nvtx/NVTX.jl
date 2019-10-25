module NVTX

using ..CUDAnative: libnvtx
using ..CUDAdrv: CUstream, CUdevice, CUcontext, CUevent

using CEnum

include("libnvtx_common.jl")
include("libnvtx.jl")

include("highlevel.jl")

end
