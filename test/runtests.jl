using AttractorModels
using StableRNGs
using CairoMakie
using Test

@testset "Basic" begin
    rng = StableRNG(1234)
    func,gfunc,ifunc = AttractorModels.get_attractors2(;w1=sqrt(10.0/2), w2=sqrt(45.0/2.0),wf=sqrt(5.0/2), b=-4.5,ϵ2=2.5,A0=7.0, A2=7.0,zmin=-3.7,ϕ=3π/4)
    X0 = [3.4,2.1]
    @test func(X0) ≈ -0.033074961274077636
    @test gfunc(X0) ≈ [0.005291993803852422,-0.022196974010603214]
    @test ifunc(1.0,rng) ≈ [-0.14570402099396693, 0.2636588502608179]
end

@testset "Trajectory" begin
    rng = StableRNG(1234)
    func,gfunc,ifunc = AttractorModels.get_attractors2(;w1=sqrt(10.0/2), w2=sqrt(45.0/2.0),wf=sqrt(5.0/2), b=-4.5,ϵ2=2.5,A0=7.0, A2=7.0,zmin=-3.7,ϕ=3π/4)
    nframes = 300
    Xl = AttractorModels.get_trajectories(func, gfunc, ifunc;bump_dur=3, nframes=nframes,bump_amp=0.0, σn=0.012, dt=0.5, max_width_scale=1.0, rebound=false,ntrials=1, freeze_before_bump=false,r0=1.0, b20=7.0, well_min=7.0, basin_scale_min=1.0,bump_time=50,do_save=true,zmin_f=0.001, zf0=3.7,b0=4.5,ϵ0=2.5, ϵf=1.0, rng=rng)
    @test length(Xl[]) == nframes + 2
    @show Xl[][1]    
    @test Xl[][1] ≈  Float32[-5.145704, -4.736341, -8.4]
    @show Xl[][end-1]
    @test Xl[][end-1] ≈  Float32[6.9926105, -13.02594, -8.4]
end

@testset "Animation" begin
    outfile = "my_cool_movie_test.mp4" 
    rng = StableRNG(1234)
    func,gfunc,ifunc = AttractorModels.get_attractors2(;w1=sqrt(10.0/2), w2=sqrt(45.0/2.0),wf=sqrt(5.0/2), b=-4.5,ϵ2=2.5,A0=7.0, A2=7.0,zmin=-3.7,ϕ=3π/4)
    fig = AttractorModels.animate_manifold(func, gfunc, ifunc;bump_dur=3, nframes=100,bump_amp=0.0, σn=0.012, dt=0.5, max_width_scale=1.0, rebound=false,ntrials=5, freeze_before_bump=false,r0=1.0, b20=7.0, well_min=7.0, basin_scale_min=1.0,bump_time=50,do_save=true,zmin_f=0.001, zf0=3.7,b0=4.5,ϵ0=2.5, ϵf=1.0,fname="sometest.jld2", do_record=true, animation_filename=outfile, rng=rng)
     @test isfile(outfile)
     rm(outfile)
end



