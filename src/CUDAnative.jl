module CUDAnative

using CUDAapi
using CUDAdrv

using LLVM
using LLVM.Interop

using Adapt
using TimerOutputs
using DataStructures

using Libdl


## global state

# version compatibility
const target_support = Ref{Vector{VersionNumber}}()
const ptx_support = Ref{Vector{VersionNumber}}()

const depsfile = joinpath(dirname(dirname(@__FILE__)), "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("CUDAnative is not properly installed. Please run Pkg.build(\"CUDAnative\")")
end

function version()::VersionNumber
    isa(cuda_toolkit_version, VersionNumber) ? cuda_toolkit_version :
    cuda_toolkit_version[] !== nothing       ? cuda_toolkit_version[] :
    error("You should only call this function if CUDAnative.jl has successfully initialized")
end


## source code includes

include("utils.jl")

# needs to be loaded _before_ the compiler infrastructure, because of generated functions
include("device/tools.jl")
include("device/pointer.jl")
include("device/array.jl")
include("device/cuda.jl")
include("device/llvm.jl")
include("device/runtime.jl")

include("init.jl")

include("compiler.jl")
include("execution.jl")
include("exceptions.jl")
include("reflection.jl")

include("deprecated.jl")


## initialization

const __initialized__ = Ref(false)
functional() = __initialized__[]

function __init__()
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false"))
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional()
        verbose && @warn "CUDAnative.jl did not initialize because CUDAdrv.jl failed to"
        return
    end

    buildlog = joinpath(dirname(@__DIR__), "deps", "build.log")
    try
        ## delayed detection

        # if we're not using BinaryBuilder, we can't be sure of everything at build-time
        if !precompiling
            if !use_binarybuilder
                @warn """Automatic installation of the CUDA toolkit failed; see $buildlog for more details
                         or call Pkg.build("CUDAnative") to try again. Otherwise, you will need to install CUDA and make sure it is discoverable."""
            end
        end

        if isa(cuda_toolkit_version, Ref)
            # parse the ptxas version string
            verstr = withenv("LANG"=>"C") do
                read(`$ptxas --version`, String)
            end
            m = match(r"\brelease (?<major>\d+).(?<minor>\d+)\b", verstr)
            m !== nothing || error("could not parse CUDA version info (\"$verstr\")")

            cuda_toolkit_version[] = VersionNumber(parse(Int, m[:major]), parse(Int, m[:minor]))
        end

        if isa(libcudadevrt, Ref)
            path = if haskey(ENV, "JULIA_CUDA_DEVICERT")
                ENV["JULIA_CUDA_DEVICERT"]
            else
                paths = [
                    "/usr/lib/x86_64-linux-gnu/libcudadevrt.a",
                    "/usr/local/cuda/targets/x86_64-linux/lib/libcudadevrt.a",
                    "/opt/cuda/targets/x86_64-linux/lib/libcudadevrt.a",
                    "/usr/local/cuda/targets/aarch64-linux/lib/libcudadevrt.a",
                    "opt/cuda/targets/aarch64-linux/lib/libcudadevrt.a",
                    "/usr/local/cuda/targets/x86_64-linux/lib/libcudadevrt.a",
                    "/opt/cuda/targets/x86_64-linux/lib/libcudadevrt.a",
                ]
                index = findfirst(isfile, paths)
                index === nothing ? nothing : paths[index]
            end
            if path === nothing || !isfile(path)
                error("Could not find the CUDA device runtime (libcudadevrt). Please specify using the JULIA_CUDA_DEVICERT environment variable.")
            end

            libcudadevrt[] = path
        end

        if isa(libdevice, Ref)
            path = if haskey(ENV, "JULIA_CUDA_LIBDEVICE")
                ENV["JULIA_CUDA_LIBDEVICE"]
            else
                paths = [
                    "/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc",
                    "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                    "/opt/cuda/nvvm/libdevice/libdevice.10.bc",
                ]
                index = findfirst(isfile, paths)
                index === nothing ? nothing : paths[index]
            end
            if path === nothing || !isfile(path)
                error("Could not find the CUDA device library (libdevice). Please specify using the JULIA_CUDA_LIBDEVICE environment variable.")
            end

            libdevice[] = path
        end

        check_deps()

        if !precompiling
            if version() < v"9"
                @warn "CUDAnative.jl only supports CUDA 9.0 or higher (your toolkit provides CUDA $(version()))"
            elseif version() > CUDAdrv.version()
                if use_binarybuilder
                    @warn """You are using CUDA toolkit $(version()) with a driver that only supports up to $(CUDAdrv.version()).
                             Rebuilding CUDAnative might fix this, try Pkg.build(\"CUDAnative\")."""
                else
                    @warn """You are using CUDA toolkit $(version()) with a driver that only supports up to $(CUDAdrv.version()).
                             It is recommended to upgrade your driver, or switch to automatic installation of CUDA."""
                end
            end
        end


        ## target support

        # LLVM.jl

        llvm_version = LLVM.version()
        llvm_targets, llvm_isas = llvm_support(llvm_version)


        # Julia

        julia_llvm_version = Base.libllvm_version
        if julia_llvm_version != llvm_version
            error("LLVM $llvm_version incompatible with Julia's LLVM $julia_llvm_version")
        end


        # CUDA

        cuda_targets, cuda_isas = cuda_support(CUDAdrv.version(), version())

        target_support[] = sort(collect(llvm_targets ∩ cuda_targets))
        isempty(target_support[]) && error("Your toolchain does not support any device target")

        ptx_support[] = sort(collect(llvm_isas ∩ cuda_isas))
        isempty(ptx_support[]) && error("Your toolchain does not support any PTX ISA")

        @debug("CUDAnative supports devices $(verlist(target_support[])); PTX $(verlist(ptx_support[]))")


        ## actual initialization

        __init_compiler__()

        CUDAdrv.apicall_hook[] = maybe_initialize

        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        if !silent && !precompiling
            if verbose
                @error "CUDAnative.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "CUDAnative.jl failed to initialize, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

verlist(vers) = join(map(ver->"$(ver.major).$(ver.minor)", sort(collect(vers))), ", ", " and ")

function llvm_support(version)
    @debug("Using LLVM v$version")

    # https://github.com/JuliaGPU/CUDAnative.jl/issues/428
    if version >= v"8.0" && VERSION < v"1.3.0-DEV.547"
        error("LLVM 8.0 requires a newer version of Julia")
    end

    InitializeAllTargets()
    haskey(targets(), "nvptx") ||
        error("""
            Your LLVM does not support the NVPTX back-end.

            This is very strange; both the official binaries
            and an unmodified build should contain this back-end.""")

    target_support = sort(collect(CUDAapi.devices_for_llvm(version)))

    ptx_support = CUDAapi.isas_for_llvm(version)
    push!(ptx_support, v"6.0") # JuliaLang/julia#23817
    ptx_support = sort(collect(ptx_support))

    @debug("LLVM supports devices $(verlist(target_support)); PTX $(verlist(ptx_support))")
    return target_support, ptx_support
end

function cuda_support(driver_version, toolkit_version)
    @debug("Using CUDA driver v$driver_version and toolkit v$toolkit_version")

    # the toolkit version as reported contains major.minor.patch,
    # but the version number returned by libcuda is only major.minor.
    toolkit_version = VersionNumber(toolkit_version.major, toolkit_version.minor)
    if toolkit_version > driver_version
        @warn("""CUDA $(toolkit_version.major).$(toolkit_version.minor) is not supported by
                 your driver (which supports up to $(driver_version.major).$(driver_version.minor))""")
    end

    driver_target_support = CUDAapi.devices_for_cuda(driver_version)
    toolkit_target_support = CUDAapi.devices_for_cuda(toolkit_version)
    target_support = sort(collect(driver_target_support ∩ toolkit_target_support))

    driver_ptx_support = CUDAapi.isas_for_cuda(driver_version)
    toolkit_ptx_support = CUDAapi.isas_for_cuda(toolkit_version)
    ptx_support = sort(collect(driver_ptx_support ∩ toolkit_ptx_support))

    @debug("CUDA driver supports devices $(verlist(driver_target_support)); PTX $(verlist(driver_ptx_support))")
    @debug("CUDA toolkit supports devices $(verlist(toolkit_target_support)); PTX $(verlist(toolkit_ptx_support))")

    return target_support, ptx_support
end

end
