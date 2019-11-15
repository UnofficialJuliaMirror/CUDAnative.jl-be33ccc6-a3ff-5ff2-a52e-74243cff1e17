using BinaryProvider
using CUDAdrv

# Parse some basic command-line arguments
const verbose = "--verbose" in ARGS
const prefix = Prefix(get([a for a in ARGS if a != "--verbose"], 1, joinpath(@__DIR__, "usr")))

# online sources we can use
const bin_prefix = "https://github.com/JuliaGPU/CUDABuilder/releases/download/v0.1.3"
const resources = Dict(
    v"9.0" =>
        Dict(
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDA.v9.0.176-0.1.3.x86_64-linux-gnu.tar.gz", "50c792a89d9b6168cd64b2ac655774a3a60300c15cb6fe09da7bb09f8d4a81cf"),
            Windows(:x86_64) => ("$bin_prefix/CUDA.v9.0.176-0.1.3.x86_64-w64-mingw32.tar.gz", "a8ec851546f2d3398010b352ff7cddfdb21a17cde388c1596b9539d38bb11283"),
            MacOS(:x86_64) => ("$bin_prefix/CUDA.v9.0.176-0.1.3.x86_64-apple-darwin14.tar.gz", "9aee98c2a286300b0e1705c7985379730f951ee408bf032a3b532c8ac0a8f369"),
        ),
    v"9.2" =>
        Dict(
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDA.v9.2.148-0.1.3.x86_64-linux-gnu.tar.gz", "efd70834cb9d2966560cd29ab4fd385f275f83b8f2c2b12c2685e9f21b0c6c88"),
            Windows(:x86_64) => ("$bin_prefix/CUDA.v9.2.148-0.1.3.x86_64-w64-mingw32.tar.gz", "0b375187c9173c2326c7118856143ef0feafd97702f4fc9dfe83037900ad3c58"),
            MacOS(:x86_64) => ("$bin_prefix/CUDA.v9.2.148-0.1.3.x86_64-apple-darwin14.tar.gz", "1c374b679bab84065d09e66eebfa1d7ba207d92115550f534152618ac19b7204"),
        ),
    v"10.0" =>
        Dict(
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDA.v10.0.130-0.1.3.x86_64-linux-gnu.tar.gz", "270119829fb6ae8aabbb8c517c9538889865646c177ab80a8c3bf04ff7343f49"),
            Windows(:x86_64) => ("$bin_prefix/CUDA.v10.0.130-0.1.3.x86_64-w64-mingw32.tar.gz", "ca938e8ec4f31581627dd7a11a92165fd472c2551d79553daf058a10fcb614f2"),
            MacOS(:x86_64) => ("$bin_prefix/CUDA.v10.0.130-0.1.3.x86_64-apple-darwin14.tar.gz", "d1cb8851484a7635fd51cd0da6c82f79f5a65572b5e6a33c855d2ea98f796282"),
        ),
    v"10.1" =>
        Dict(
            Linux(:x86_64, libc=:glibc) => ("$bin_prefix/CUDA.v10.1.243-0.1.3.x86_64-linux-gnu.tar.gz", "ba6086c0a3df31d419abba7c18feadf67a832197c29d6e12643187cbf7b92464"),
            Windows(:x86_64) => ("$bin_prefix/CUDA.v10.1.243-0.1.3.x86_64-w64-mingw32.tar.gz", "532d99ab03d2360718a3b299afef087eaf09243e69033793bd057f3f9682e1b1"),
            MacOS(:x86_64) => ("$bin_prefix/CUDA.v10.1.243-0.1.3.x86_64-apple-darwin14.tar.gz", "34bea1c4c4b846aa2757cca66a1b8fc37cfe6abcddb0a2995da6788aa792cf79"),
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

    use_binarybuilder = parse(Bool, get(ENV, "JULIA_CUDA_USE_BINARYBUILDER", "true"))
    if use_binarybuilder
        if try_binarybuilder()
            @assert !unsatisfied()
            return
        end
    end

    do_fallback()

    return
end

verlist(vers) = join(map(ver->"$(ver.major).$(ver.minor)", sort(collect(vers))), ", ", " and ")

# download CUDA using BinaryBuilder
function try_binarybuilder()
    @info "Trying to provide CUDA using BinaryBuilder"

    # CUDA version selection
    cuda_version = if haskey(ENV, "JULIA_CUDA_VERSION")
        @warn "Overriding CUDA version to $(ENV["JULIA_CUDA_VERSION"])"
        VersionNumber(ENV["JULIA_CUDA_VERSION"])
    elseif CUDAdrv.functional()
        driver_capability = CUDAdrv.version()
        @info "Detected CUDA driver compatibility $(driver_capability)"

        # CUDA drivers are backwards compatible
        supported_versions = filter(ver->ver <= driver_capability, keys(resources))
        if isempty(supported_versions)
            @warn("""Unsupported version of CUDA; only $(verlist(keys(resources))) are available through BinaryBuilder.
                     If your GPU and driver supports it, you can force a different version with the JULIA_CUDA_VERSION environment variable.""")
            return false
        end

        # pick the most recent version
        maximum(supported_versions)
    else
        @warn("""Could not query CUDA driver compatibility. Please fix your CUDA driver (make sure CUDAdrv.jl works).
                 Alternatively, you can force a CUDA version with the JULIA_CUDA_VERSION environment variable.""")
        return false
    end
    @info "Selected CUDA $cuda_version"

    if !haskey(resources, cuda_version)
        @warn("Requested CUDA version is not available through BinaryBuilder.")
        return false
    end
    download_info = resources[cuda_version]

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
    @warn "Could not download CUDA dependencies; assuming they will be available at run time"

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
