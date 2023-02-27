x1,y1 = (-5.0, -5.0)
using LinearAlgebra
using Makie
using GLMakie
using Makie: Point2f0
using Colors
using StableRNGs
using Dierckx
using Statistics
using MultivariateStats

#TODO: Extend this to higher dimenions
#      First do 3D,i.e. 3 units instead of 2. In that case, we can't really illustrate the potential, but we can plot the 3D trajectories along with the surface of the potenial.

b0 = 4.0
x1,y1 = (-5.0, -5.0)
w1 = 9.0
x0n,y0n = (-5.0, -5.0)
z(d) = cos(2π*d/10)
z(x,y) = z(sqrt((x-5.0)^2+(y-5.0)^2))
gz(x,y) = (d = sqrt((x-5.0)^2+(y-5)^2); (-2*sin(2π*d/10)*(2π/10)/d).*[x,y])

aa1 = 2.0
flat_bottom(x,y,b=b0,aa=aa1,zmin=-Inf) = max(-b*exp(-((x-x0n)^aa1 + (y-y0n)^aa1)/3.5), zmin)

function g_flat_bottom(x,y,b,aa,zmin)
    z = flat_bottom(x,y,b,aa,-Inf)
    if z < zmin
        return [0.0, 0.0]
    end
    g = aa/3.5*z*[(x-x0n)^(aa-1), (y-y0n)^(aa-1)]
end

#f1(x,y,b=b0) = 7.0*exp(-((x-x1)^2 + (y-y1)^2)./7.0) - b*exp(-((x-x0n)^aa1 + (y-y0n)^aa1)/2.5)
#f1(x,y,b=b0) = 7.0*exp(-((x-x1)^2 + (y-y1)^2)./7.0)  + flat_bottom(x,y,b,aa1,-3.5)
#g1(x,y,b=b0) = [2*exp(-((x-x1)^2 + (y-y1)^2)/7.0)*(x-x1) - aa1*b/2.5*exp(-((x-x0n)^aa1 + (y-y0n)^aa1)/2.5)*(x-x0n)^(aa1-1),
#                2*exp(-((x-x1)^2 + (y-y1)^2)/7.0)*(y-y1)- aa1*b/2.5*exp(-((x-x0n)^aa1 + (y-y0n)^aa1)/2.5)*(y-y0n)^(aa1-1)]
#
function f1(x,y,b=b0,zmin=-3.2)
    z1 = 7.0*exp(-((x-x1)^2 + (y-y1)^2)./9.0)
    z2 = flat_bottom(x,y,b,aa1,-Inf)
    if -b < zmin
        d0 = sqrt(-3.5*log(zmin/-b))
        if  sqrt((x-x1)^2 + (y-y1)^2) < d0
            z1 = 7.0*exp(-d0^2/9.0)
            z2 = zmin 
        end
    end
    return z1 + z2
end

function g1(x,y,b=b0,zmin=-3.2)
    z1 = 7.0*exp(-((x-x1)^2 + (y-y1)^2)./9.0) 
    z2 = flat_bottom(x,y,b,aa1,-Inf)
    if z2 < zmin 
        return [0.0, 0.0]
    end
    gxy = 2*z1/9.0*[x-x1, y-y1]
    gxy .+= g_flat_bottom(x,y,b,aa1,-Inf)
    gxy
end

x2,y2 = (7.0, -13.0)
w2 = 60.0
f2(x,y) = -4.0*(exp(-((x - x2)^2 + (y-y2)^2)./w2))
g2(x,y) = -2.0*4.0/w2*exp(-((x - x2)^2 + (y-y2)^2)./w2).*[x-x2, y-y2]


f(x,y,b=b0) = f1(x,y,b) + f2(x,y)
g(x,y,b=b0) = g1(x,y,b) + g2(x,y)
g(xy::Point2f0) = g(xy[1], xy[2]) 

function run_model(impulse::Observable{Float32}, stop::Observable{Bool}, σn::Float64=0.01)
    pos0 = Observable([Point2f0(x1, y1)])
    xx = range(-10, stop=20.0, length=100)
    yy = range(-25, stop=5.0, length=100) 
    fig = Figure(resolution=(1000,1000))
    ax = Axis(fig[1,1])
    cf = contour!(ax, xx, yy, [f(_x,_y) for _x in xx, _y in yy], levels=15)
    cb = Colorbar(fig[1,2], cf, label="Potential")
    scatter!(ax,pos0) 
    fps = 60.0
    display(fig)
    @async for i in 1:5000
        pos = pos0[]
        x,y = pos[1][1], pos[1][2]
        Δ = g(x,y)
        q = norm(Δ)
        Δ ./= q
        a = min(max(0.01/q, 0.01), 1.0)
        qn = Point2f0(σn*randn(2))
        pos = pos .+ (Point2f0(a*Δ) + Point2f0(impulse[]) + qn)
        pos0[] = pos
        if stop[]
            break
        end
        sleep(1.0/fps)
        yield()
     end
    fig
end

function bump!(impulse::Observable{T}, x=0.05, dt=0.1) where T <: Real
    p = impulse[]
    impulse[] = p + x
    sleep(dt)
    impulse[] = p
end


function plot_potential(args...)
    fig = Figure(resolution=(1000,1000))
    ax = Axis(fig[1,1])
    cf = plot_potential!(ax, args...)
    cb = Colorbar(fig[1,2], cf, label="Potential")
    fig, ax
end

function plot_potential!(ax, n=100)
    xx = range(-10, stop=20.0, length=100)
    yy = range(-25, stop=5.0, length=100) 
    cf = contour!(ax, xx, yy, [f(_x,_y) for _x in xx, _y in yy], levels=15)
end

function run_model(;σn::Float64=0.01, bump_amp=0.01, bump_time=20, bump_dur=2, nframes=100,x0=x1, y0=y1,fname="trajectory.mp4", rseed=UInt32(1236),ntrials=1, no_noise_after_bump=false)
    RNG = StableRNG(rseed)
    xx = range(-10, stop=20.0, length=100)
    yy = range(-25, stop=5.0, length=100) 
    fig = Figure(resolution=(1000,1000))
    ax = Axis(fig[1,1])
    colors = ax.palette.color[]
    cf = contour!(ax, xx, yy, [f(_x,_y) for _x in xx, _y in yy], levels=15)
    cb = Colorbar(fig[1,2], cf, label="Potential")
    sc = scatter!(ax,[x0], [y0]) 
    impulse = 0.0
    curvex = Float64[]
    curvey = Float64[]
    record(fig, fname, 1:ntrials*nframes) do i
        kk = rem(i-1, nframes) + 1
        if kk == 1
            x0 = 0.1*randn(RNG) + x1
            y0 = 0.1*randn(RNG) + y1
            #indcate trial start
            push!(curvex, NaN)
            push!(curvey, NaN)
        end
        if bump_time <= kk < bump_time+bump_dur
            j = kk-bump_time
            if  j < bump_dur/2
                b = b0 - 2*(b0 - bump_amp)*j/bump_dur 
            else
                b = bump_amp - 2*(bump_amp-b0)*(j-div(bump_dur,2))/bump_dur 
            end
            sc.color[] = colors[2]
        else
            b = b0 
            sc.color[] = colors[1]
        end
        cf[3][] = [f(_x,_y,b) for _x in xx, _y in yy]
        Δ = g(x0,y0,b)
        q = norm(Δ)
        #Δ ./= q
        #a = min(max(0.01/q, 0.01), 1.0)
        a = 1.0
        if kk > bump_time + bump_dur && no_noise_after_bump
            qn = fill(0.0,2)
        else
            qn = σn*randn(RNG, 2)
        end
        x0 += a*Δ[1]+qn[1]
        y0 += a*Δ[2]+qn[2]
        sc[1] = [x0]
        sc[2] = [y0]
        push!(curvex, x0)
        push!(curvey, y0)
     end
     curvex, curvey
end

function compute_pathlength(curves::Vector{Matrix{Float64}}, bump_time::Int64)
    nt = length(curves)
    L = fill(NaN, nt)
    for i in 1:nt
        curve = curves[i]
        l = 0.0
        for j in bump_time:size(curve, 1)
            x,y = curve[j,:]

            d = sqrt((x - x2)^2 + (y-y2)^2)
            if d < 2.0
                L[i] = l
                break
            else 
                l += sum(abs2, curve[j,:] - curve[j-1,:])
            end
        end
    end
    L
end

function plot_figure(curvex::Vector{Float64}, curvey::Vector{Float64},idx0=30)
    # create vector of matrices, one for each trial
    pidx = findall(isnan, curvex)
    curves = [[curvex[pp0+1:pp1-1] curvey[pp0+1:pp1-1]] for (pp0,pp1) in zip(pidx[1:end-1], pidx[2:end])]
    w = 30*72/2.5
    h = w
    fig = Figure(resolution=(w,h))
    lg1 = GridLayout()
    fig[1,1] = lg1
    ax1 = Axis3(lg1[1,1])
    lg2 = GridLayout() 
    fig[2,1] = lg2
    lg3 = lg2[1,2]
    ax2 = Axis(lg3[1,1])
    ax22 = Axis(lg3[2,1])
    linkxaxes!(ax2, ax22)
    ax3 = Axis(lg2[1,1])
    ax4 = Axis(lg2[1,3])
    colsize!(lg2, 3, Relative(0.1))
    xx = range(-10, stop=20.0, length=200)
    yy = range(-25, stop=5.0, length=200)
    colormap = :bwr
    vidx = (x1-0.1*w1) .< xx .< (x2+0.1*w1)
    vidy = (y2-0.1*w2) .< yy .< (y1+0.1*w2)
    sf = contour!(ax1, xx, yy, f; levels=15, colormap=colormap, colorrange=(-2.5, 2.5))
    contour!(ax3, xx[vidx], yy[vidy], f; levels=15, colormap=colormap, colorrange=(-2.5, 2.5))
    cb = Colorbar(lg1[1,2], sf, label="Potential",labelsize=24, ticklabelsize=16,ticklabelsvisible=false)
    #lines!(ax1, curvex[idx0:end], curvey[idx0:end], fill(0.0, length(curvex)-idx0+1), color="black")
    ff(x,y) = f(x,y) + 10.0
    surface!(ax1, xx, yy, ff, colormap=colormap)
    # for the 3D lines, we first interpolate, so that we can generate higher resolution lines
    sidx = 0
    eeidx = fill(0, length(curves))
    cidx = [1, 4,7]
    path_length = fill(0.0, length(curves))
    for (ii,curve) in enumerate(curves)
        d = sqrt.(dropdims(sum(abs2, curve[idx0:end,:] .- permutedims(repeat([x2,y2],1,1)),dims=2),dims=2))
        _eeidx = findfirst(d .< 0.01*sqrt(w2))
        if _eeidx != nothing
            eeidx[ii] = _eeidx
        end
        spl = ParametricSpline(permutedims(curve[idx0:end,:], [2,1]))
        ssidx = searchsortedfirst(range(extrema(spl.t)...;length=10*length(spl.t)), (sidx-1)/size(curve,1))
        curvep = Dierckx.evaluate(spl, range(extrema(spl.t)...;length=10*length(spl.t)))
        d = sqrt.(dropdims(sum(abs2, curvep .- repeat([x2,y2],1,1),dims=1),dims=1))
        _eeidx = findfirst(d .< 0.05*sqrt(w2))
        path_length[ii] = sum(sqrt.(sum(abs2, diff(curvep[:,1:_eeidx],dims=2),dims=1)))
        lines!(ax1, curvep[1,:], curvep[2,:], ff.(curvep[1,:], curvep[2,:]), color="black", linewidth=2.0)
        lines!(ax1, curvep[1,:], curvep[2,:], fill(0.0, size(curvep,2)), color="black")
        if ii in cidx
            jj = findfirst(cidx.==ii)
            lines!(ax3, curvep[1,1:_eeidx], curvep[2,1:_eeidx], color=ax3.palette.color[][1+jj], linewidth=2.0)
            scatter!(ax3, curvep[1,1:1], curvep[2,1:1], markersize=20px, color=ax3.palette.color[][1])
        end
    end
    tidx = findall(eeidx.!=0)
    @show sum(eeidx .== 0)
    σ = fill(0.0, size(curves[1],1))
    r² = fill(.0, size(curves[1],1))
    lrt = log.(eeidx[tidx])
    lrt .-= mean(lrt)
    for i in 1:size(curves[1],1)
        _curves = cat([curve[i:i,:] for curve in curves]...,dims=1)
        # regress "reaction time"
        X = _curves[tidx,:]
        X .-= mean(X,dims=1)
        β = llsq(X, lrt;bias=false)
        prt = X*β
        r²[i] = 1.0 - sum(abs2, prt .- lrt)./sum(abs2, lrt)
        Σ = cov(_curves)
        u,s,v = svd(Σ)
        σ[i] = sum(s)
    end
    sidx = argmax(σ)
    ax1.xticksvisible = false
    ax1.yticksvisible = false
    ax1.zticksvisible = false
    ax1.xticklabelsvisible = false
    ax1.yticklabelsvisible = false
    ax1.zticklabelsvisible = false
    ax1.xlabel = "Neuron 1"
    ax1.ylabel = "Neuron 2"
    ax1.xlabelsize=24
    ax1.ylabelsize=24


    for _ax in [ax2,ax22]
        _ax.xgridvisible = false
        _ax.ygridvisible = false
        _ax.topspinevisible = false
        _ax.rightspinevisible = false
        _ax.xlabelsize=24
        _ax.ylabelsize=24
        _ax.xticklabelsvisible = false
        _ax.yticklabelsvisible = false
    end
    ax2.ylabel = "Variance"
    ax22.ylabel = "rt-pred"
    ax22.xlabel = "Time"
    ax22.yticklabelsvisible = true
    ax3.xgridvisible = false
    ax3.ygridvisible = false
    ax3.topspinevisible = false
    ax3.rightspinevisible = false
    ax3.xlabelsize = 24
    ax3.xlabel = "Neuron 1"
    ax3.ylabelsize=24
    ax3.ylabel = "Neuron 2"
    ax3.xticklabelsvisible = false
    ax3.yticklabelsvisible = false
    bidx = idx0:(idx0+minimum(eeidx)-1)
    tt = range(0.0, stop=1.0, length=minimum(eeidx[tidx]))
    lines!(ax2, tt, σ[idx0:idx0+minimum(eeidx[tidx])-1])
    lines!(ax22, tt, r²[idx0:idx0+minimum(eeidx[tidx])-1])
    rowsize!(fig.layout, 2, Relative(0.3))
    pcolors = fill(RGB(0.0, 0.0, 0.0), length(path_length))
    pcolors[cidx] .= ax3.palette.color[][2:4]
    scatter!(ax4, 1:length(path_length), path_length,color=pcolors, markersize=15px)
    ax4.xticklabelsvisible = false
    ax4.xgridvisible = false
    ax4.ygridvisible = false
    ax4.topspinevisible = false
    ax4.rightspinevisible = false
    ax4.ylabel = "Path length"
    fig
end

impulse = Observable(0.0f0)
stop = Observable(false)
#fig = run_model(impulse, stop)
