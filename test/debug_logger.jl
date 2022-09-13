import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
"""
Debug formatter, highlight NLSolversBase. To use:
```
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do 
    @test ...
end
```
"""
function locfmt(level::LogLevel, _module, group, id, file, line)
    @nospecialize
    if repr(_module) == "FiniteDiff"
        color = :green
    elseif repr(_module) == "Main"
        color = :176
    elseif repr(_module) ==  "MechGlueDiffEqBase"
        color = :magenta
    else
        color = :blue
    end
    prefix = string(level == Logging.Warn ? "Warning" : string(level), ':')
    suffix::String = ""
    Logging.Info <= level < Logging.Warn && return color, prefix, suffix
    _module !== nothing && (suffix *= "$(_module)")
    if file !== nothing
        _module !== nothing && (suffix *= " ")
        suffix *= Base.contractuser(file)::String
        if line !== nothing
            suffix *= ":$(isa(line, UnitRange) ? "$(first(line))-$(last(line))" : line)"
        end
    end
    !isempty(suffix) && (suffix = "@ " * suffix)
    return color, prefix, suffix
end
nothing