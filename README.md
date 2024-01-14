# AttractorModels.jl
Toy models for various attractor landscapes written in julia

![CI](https://github.com/grero/AttractorModels.jl/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/grero/AttractorModels.jl/branch/master/graph/badge.svg?token=71vAhb76D6)](https://codecov.io/gh/grero/AttractorModels.jl)

## Installation

First, add the NeuralCodingRegistry, in which this package is registered, as a registry in julia. Then, simply add the package as you would a normal Julia package.

```julia
pkg> registry add https://github.com/grero/NeuralCodingRegistry.jl.git
pkg> add AttractorModels, Makie, GLMakie

```

## Usage

We'll create a 2D manifold embedded in 3D, consisting of two stable states; one at coordinates (-5,-5) which consists of a raised, shallow basin, and one at coordinates (7,-13) which consists of a single deep well. 
Then, we'll generate trajectories from this manifold, where we'll introduce a "go-cue" which will temporarily raise the basin, allowing a state confied to the manifold to escape the basin. 

```juila
using AttractorModels
using Makie, GLMakie
func,gfunc,ifunc = AttractorModels.get_attractors2(;A0=7.0, # height of the raised basin
                                                    A2=7.0, # depth of the well
                                                    w1=sqrt(10.0/2), # width of the raised part
                                                    wf=sqrt(5.0/2),  # width of the basin
                                                    w2=sqrt(45.0/2.0), # width of the well 
                                                    b=-4.5, # depth of the basin
                                                    ϵ2=2.5, # eccentricity of the basin
                                                    zmin=-3.5, # the floor of the well
                                                    ϕ=3π/4 # the orientation of the basin
                                                    )

# now we animate it
# file to save the curves to
curve_data_file = "curves.jld2"
fig = AttractorModels.animate_manifold(func, gfunc, ifunc;bump_dur=3, # duration of the go-cue bump
                                                          bump_amp=0.0, # basin damping factor during go-cue
                                                          nframes=300, # number of frames
                                                          ntrials=500, # number trials, i.e. curves
                                                          σn=0.0525, # level of gaussian noise added
                                                          dt=0.5,  # time step 
                                                          max_width_scale=1.0, # expansion of basin during bump; 1.0 means no expansion
                                                          r0=1.0, # radius of the initial state
                                                          b20=7.0, # initial depth of the end state well
                                                          well_min=7.0, # final depth of the end state well
                                                          bump_time=50, # time of go-cue bump onset 
                                                          zmin_f=0.001, # final basin depth
                                                          zf0=3.5, b0=4.5, # initial basin parameters 
                                                          ϵ0=2.5, ϵf=1.0, # initial eccentricities
                                                          do_record=true,
                                                          animation_filename="test_movie.mp4",
                                                          fps=60.0,
                                                          do_save=true,
                                                          fname=curve_data_file)

```
