x1,y1 = (-5.0, -5.0)
using LinearAlgebra
using Makie
using GLMakie
using Makie: Point2f0
using StableRNGs


b0 = 3.2
x1,y1 = (-5.0, -5.0)
x0n,y0n = (-5.0, -5.0)
z(d) = cos(2π*d/10)
z(x,y) = z(sqrt((x-5.0)^2+(y-5.0)^2))
gz(x,y) = (d = sqrt((x-5.0)^2+(y-5)^2); (-2*sin(2π*d/10)*(2π/10)/d).*[x,y])

f1(x,y,b=b0) = 7.0*exp(-((x-x1)^2 + (y-y1)^2)./7.0) - b*exp(-((x-x0n)^4 + (y-y0n)^4)/2.5)
g1(x,y,b=b0) = [2*exp(-((x-x1)^2 + (y-y1)^2)/7.0)*(x-x1) - 4*b/2.5*exp(-((x-x0n)^4 + (y-y0n)^4)/2.5)*(x-x0n)^3,
            2*exp(-((x-x1)^2 + (y-y1)^2)/7.0)*(y-y1)- 4*b/2.5*exp(-((x-x0n)^4 + (y-y0n)^4)/2.5)*(y-y0n)^3]

x2,y2 = (5.0, -10.0)
f2(x,y) = -3.0*(exp(-((x - x2)^2 + (y-y2)^2)./60.0))
g2(x,y) = -2.0*3.0/60.0*exp(-((x - x2)^2 + (y-y2)^2)./60.0).*[x-x2, y-y2]


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

impulse = Observable(0.0f0)
stop = Observable(false)
#fig = run_model(impulse, stop)
