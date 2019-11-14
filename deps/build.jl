using BinaryProvider
using CUDAdrv

# Parse some basic command-line arguments
const verbose = "--verbose" in ARGS
const prefix = Prefix(get([a for a in ARGS if a != "--verbose"], 1, joinpath(@__DIR__, "usr")))

# online sources we can use
const bin_prefix = "https://github.com/JuliaGPU/CUDABuilder/releases/download/v0.1.1"
const resources = Dict(
    v"10.1" =>
        Dict(
            MacOS(:x86_64) => ("$bin_prefix/CUDA.v10.1.243.x86_64-apple-darwin14.tar.gz", "d8e2bad9bd1fcd93aef48e16edcdbe078c8517301946382e2348a12bfb2fbdd3"),
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDA.v10.1.243.x86_64-linux-gnu.tar.gz", "2b0abbf2d1b198663038c07f6cd25eda2f26c80ac8e4b6632ebbd8123a08e8c4"),
            Windows(:x86_64) => ("$bin_prefix/CUDA.v10.1.243.x86_64-w64-mingw32.tar.gz", "13354a23a0f5f158ce2a79798e91741bf2fc84fc5e1ba682c220d5dfa247a663"),
        ),
    v"10.0" =>
        Dict(
            MacOS(:x86_64) => ("$bin_prefix/CUDA.v10.0.130.x86_64-apple-darwin14.tar.gz", "8e76201a82b1ddf695bf07b3d2bf66491992563f86da43242e29561c3406f6f0"),
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDA.v10.0.130.x86_64-linux-gnu.tar.gz", "605a2b5f1c8b840e4e0898725780010a5b5d8e776a44b1b19097866f0e29ed5e"),
            Windows(:x86_64) => ("$bin_prefix/CUDA.v10.0.130.x86_64-w64-mingw32.tar.gz", "3989c6019128e7c814b36da30151390223f596c3e17779d932284ced9bcc8f6c"),
        ),
    v"9.2" =>
        Dict(
            MacOS(:x86_64) => ("$bin_prefix/CUDA.v9.2.148.x86_64-apple-darwin14.tar.gz", "19338ae2f97c7d840b49d5095230e03e429511c3750cbfb8c6482ce0570992d3"),
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDA.v9.2.148.x86_64-linux-gnu.tar.gz", "fff8cd027e95b62c552c0225c5daeb6d751b94d984be5c3ca849d61cfef6dd34"),
            Windows(:x86_64) => ("$bin_prefix/CUDA.v9.2.148.x86_64-w64-mingw32.tar.gz", "0768b7293320ff8f4a4ae35d9ea4a02c43bdefcebc8209c94fdaff9010204de7"),
        ),
    v"9.0" =>
        Dict(
            MacOS(:x86_64) => ("$bin_prefix/CUDA.v9.0.176.x86_64-apple-darwin14.tar.gz", "d50e80e9fe58551a2092cec74b6a3fe680ad13ebf85a5f7bd6b184ef72ec1f18"),
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDA.v9.0.176.x86_64-linux-gnu.tar.gz", "120224f076be11846d7f22225ac8fa23eba82dc6356868d20814b89a8e795c8c"),
            Windows(:x86_64) => ("$bin_prefix/CUDA.v9.0.176.x86_64-w64-mingw32.tar.gz", "5cf9ca0530e09b48a578d73fef08386b74b422fcddf15e0c7c6cfebc94f11fa0"),
        ),
)

# stuff we need to resolve
const products = [
    ExecutableProduct(prefix, "nvdisasm", :nvdisasm),
    ExecutableProduct(prefix, "ptxas", :ptxas),
    FileProduct(prefix, "share/libdevice/libdevice.10.bc", :libdevice),
    FileProduct(prefix, Sys.iswindows() ? "lib/cudadevrt.lib" : "lib/libcudadevrt.a", :libcudadevrt)
]
unsatisfied() = any(!satisfied(p; verbose=verbose) for p in products)

const depsfile = joinpath(@__DIR__, "deps.jl")

function main()
    rm(depsfile; force=true)

    # CUDA version selection
    cuda_version = if haskey(ENV, "JULIA_CUDA_VERSION")
        # use the CUDA version as requested by the user
        @warn "Overriding CUDA version to $(ENV["JULIA_CUDA_VERSION"])"
        VersionNumber(ENV["JULIA_CUDA_VERSION"])
    elseif CUDAdrv.functional()
        # use a version of CUDA that matches the installed driver
        @debug "Detected CUDA driver compatibility $(CUDAdrv.version())"
        CUDAdrv.version()
    else
        nothing
    end

    use_binarybuilder = parse(Bool, get(ENV, "JULIA_CUDA_USE_BINARYBUILDER", "true"))
    if use_binarybuilder && cuda_version !== nothing
        if try_binarybuilder(cuda_version)
            @assert !unsatisfied()
            return
        end
    end

    do_fallback()

    return
end

verlist(vers) = join(map(ver->"$(ver.major).$(ver.minor)", sort(collect(vers))), ", ", " and ")

# download CUDA using BinaryBuilder
function try_binarybuilder(cuda_version)
    if haskey(resources, cuda_version)
        download_info = resources[cuda_version]
    else
        @warn("""Unsupported version of CUDA requested; only $(verlist(keys(resources))) are supported through BinaryBuilder.
                If your GPU and driver supports it, you can force a different version with the JULIA_CUDA_VERSION environment variable.""")
        return false
    end

    # Install unsatisfied or updated dependencies:
    dl_info = choose_download(download_info, platform_key_abi())
    if dl_info === nothing && unsatisfied()
        # If we don't have a compatible .tar.gz to download, complain.
        # Alternatively, you could attempt to install from a separate provider,
        # build from source or something even more ambitious here.
        @warn("Your platform (\"$(Sys.MACHINE)\", parsed as \"$(triplet(platform_key_abi()))\") is not supported through BinaryBuilder.")
        return false
    end

    # If we have a download, and we are unsatisfied (or the version we're
    # trying to install is not itself installed) then load it up!
    if unsatisfied() || !isinstalled(dl_info...; prefix=prefix)
        # Download and install binaries
        install(dl_info...; prefix=prefix, force=true, verbose=verbose)
    end

    # Write out a deps.jl file that will contain mappings for our products
    write_deps_file(depsfile, products, verbose=verbose)

    open(depsfile, "a") do io
        println(io)
        println(io, "const use_binarybuilder = true")
        println(io, "const cuda_toolkit_version = $(repr(cuda_version))")
    end

    return true
end

# assume that everything will be fine at run time
function do_fallback()
    @warn "Could not download CUDA; assuming it will be available at run time"

    open(depsfile, "w") do io
        println(io, "const use_binarybuilder = false")
        for p in products
            if p isa ExecutableProduct
                # executables are expected to be available on PATH
                println(io, "const $(variable_name(p)) = $(repr(basename(p.path)))")
            elseif p isa FileProduct
                # files are more tricky and need to be resolved at run time
                println(io, "const $(variable_name(p)) = Ref{Union{Nothing,String}}(nothing)")
            end
        end
        println(io, "const cuda_toolkit_version = Ref{Union{Nothing,VersionNumber}}(nothing)")
        println(io, """
            function check_deps()
                run(pipeline(`ptxas --version`, stdout=devnull))
                run(pipeline(`nvdisasm --version`, stdout=devnull))

                @assert libdevice[] !== nothing
                @assert libcudadevrt[] !== nothing
            end""")
    end

    return
end

main()
