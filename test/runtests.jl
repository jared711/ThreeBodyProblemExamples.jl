# using ThreeBodyProblemExamples
using Test

@testset "ThreeBodyProblemExamples.jl" begin
    @test !isempty(include("../01_SystemDefinition.jl"))
    @test !isempty(include("../02_Trajectories.jl"))
    @test !isempty(include("../03_ForbiddenRegion.jl"))
    @test !isempty(include("../08_QuasiPeriodicOrbits.jl"))
end
