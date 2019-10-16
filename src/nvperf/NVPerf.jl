module NVPerf

using ..CUDAnative: libnvperf

using CEnum

include("libnvperf_common.jl")
include("error.jl")

include("libnvperf.jl")

end
