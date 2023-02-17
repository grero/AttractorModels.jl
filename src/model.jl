x1,y1 = (-5.0, -5.0)
using LinearAlgebra
using Makie
using GLMakie
using Makie: Point2f0
using StableRNGs


b0 = 4.0
x1,y1 = (-5.0, -5.0)
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

function run_model(;σn::Float64=0.01, bump_amp=0.01, bump_time=20, bump_dur=2, nframes=100,x0=x1, y0=y1,fname="trajectory.mp4", rseed=UInt32(1236),ntrials=1)
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
        qn = σn*randn(RNG, 2)
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

function plot_figure(curvex::Vector{Float64}, curvey::Vector{Float64})
    # create vector of matrices, one for each trial
    pidx = findall(isnan, curvex)
    curves = [[curvex[pp0+1:pp1-1] curvey[pp0+1:pp1-1]] for (pp0,pp1) in zip(pidx[1:end-1], pidx[2:end])]
    w = 30*72/2.5
    h = w
    fig = Figure(resolution=(w,h))
    ax1 = Axis3(fig[1,1])
    xx = range(-10, stop=20.0, length=200)
    yy = range(-25, stop=5.0, length=200)
    colormap = :bwr
    sf = contour!(ax1, xx, yy, f; levels=15, colormap=colormap, colorrange=(-2.5, 2.5))
    cb = Colorbar(fig[1,2], sf, label="Potential")
    lines!(ax1, curvex, curvey, fill(0.0, length(curvex)), color="black")
    ff(x,y) = f(x,y) + 10.0
    surface!(ax1, xx, yy, ff, colormap=colormap)
    # for the 3D lines, we first interpolate, so that we can generate higher resolution lines
    for curve in curves
        spl = ParametricSpline(permutedims(curve, [2,1]))
        curvep = evaluate(spl, range(extrema(spl.t)...;length=10*length(spl.t)))

        lines!(ax1, curvep[1,:], curvep[2,:], ff.(curvep[1,:], curvep[2,:]), color="black", linewidth=2.0)
    end
    ax1.xticksvisible = false
    ax1.yticksvisible = false
    ax1.zticksvisible = false
    ax1.xticklabelsvisible = false
    ax1.yticklabelsvisible = false
    ax1.zticklabelsvisible = false
    ax1.xlabel = "Neuron 1"
    ax1.ylabel = "Neuron 2"
    fig
end

impulse = Observable(0.0f0)
stop = Observable(false)
#fig = run_model(impulse, stop)
