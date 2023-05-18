using AttractorModels
using StableRNGs
using Test

@testset "Basic" begin
    rng = StableRNG(1234)
    func,gfunc,ifunc = AttractorModels.get_attractors2(;w1=sqrt(10.0/2), w2=sqrt(45.0/2.0),wf=sqrt(5.0/2), b=-4.5,ϵ2=2.5,A0=7.0, A2=7.0,zmin=-3.7,ϕ=3π/4)
    X0 = [3.4,2.1]
    @test func(X0) ≈ -0.033074961274077636
    @test gfunc(X0) ≈ [0.005291993803852422,-0.022196974010603214]
    @test ifunc(1.0,rng) ≈ [-0.14570402099396693, 0.2636588502608179]
end



