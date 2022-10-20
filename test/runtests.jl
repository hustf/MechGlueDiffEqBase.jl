using Test, Dates
include("debug_logger.jl")

# Setup based on Cairo.jl. Avoid re-running time consuming
# tests in the same session.
test_dir_path = @__DIR__

filenumber(s) = parse(Int, join(filter!.(isdigit, collect.(s))))
function test_one_file(test_file_name)
    output_log_name = replace(test_file_name,".jl" => ".log")
    if isfile(joinpath(test_dir_path, output_log_name))
        printstyled(output_log_name, " exists.\n"; color = 176)
        @test true
    else
        printstyled(test_file_name, "\n"; color = 176)
        # Run each sample script in a separate module to avoid pollution
        s   = Symbol(test_file_name)
        mod = @eval(Main, module $s end)
        stats = @timed @eval mod include($(joinpath(test_dir_path, test_file_name)))
        statsnext = @timed @eval mod include($(joinpath(test_dir_path, test_file_name)))
        ts = Test.get_testset()
        if length(ts.results) > 0
            println("\nTest results in ", output_log_name)
        end
        output_log_name = replace(test_file_name,".jl" => ".log")
        open(joinpath(test_dir_path, output_log_name), "w") do io
            println(io, Dates.today(), " ", test_file_name)
            if length(ts.results) > 0
                for r in ts.results
                    redirect_stdout(io) do
                        Test.print_test_results(r)
                    end
                end
            end
            println(io)
            println(io, "@timed stats:")
            println(io, stats)
            println(io, "@timed stats next run:")
            println(io, statsnext)
        end
    end
end
test_files = filter(str->endswith(str,".jl") && startswith(str,"test_"), readdir(test_dir_path))
log_files = filter(str->endswith(str,".log") && startswith(str,"test_"), readdir(test_dir_path))
files_to_exclude = replace.(log_files, ".jl" => ".log")
test_files = setdiff(test_files, files_to_exclude)


@testset "Basic adaption to units (test_n,  n < 10)" begin
    @testset "test: $test_file_name" for test_file_name in test_files
        output_log_name = replace(test_file_name,".jl" => ".log")
        if filenumber(test_file_name) < 10
            test_one_file(test_file_name)
        end
    end
end

@testset "Traits-based mutable recursive ArrayPartition (test_n, 9 < n < 20)" begin
    @testset "test: $test_file_name" for test_file_name in test_files
        if 9 < filenumber(test_file_name) < 20
            test_one_file(test_file_name)
        end
    end
end

@testset "Finite differentiation (test_n, 19 < n < 30)" begin
    @testset "test: $test_file_name" for test_file_name in test_files
        if 19 < filenumber(test_file_name) < 30
            test_one_file(test_file_name)
        end
    end
end

@testset "NLSolve, Jacobian (test_n, 29 < n < 40)" begin
    @testset "test: $test_file_name" for test_file_name in test_files
        if 29 < filenumber(test_file_name) < 40
            test_one_file(test_file_name)
        end
    end
end

@testset "BoundaryValueDiffEq (test_n, 39 < n < 50)" begin
    @testset "test: $test_file_name" for test_file_name in test_files
        if 39 < filenumber(test_file_name) < 50
            test_one_file(test_file_name)
        end
    end
end

@testset "Stiff diffeq solver (test_n, 49 < n < 60)" begin
    @testset "test: $test_file_name" for test_file_name in test_files
        if 39 < filenumber(test_file_name) < 50
            test_one_file(test_file_name)
        end
    end
end
nothing
