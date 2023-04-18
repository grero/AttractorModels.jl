module AttractorModels
using JLD2
using Dierckx
using Random
using CRC32c
using ProgressMeter
using LinearAlgebra
using Makie
using Makie: Point2f0, Point3f0

function get_attractors(;w1=sqrt(5.0), w2=sqrt(7.0), wf=sqrt(3.5), A0=7.0, b=-4.0)
    Xi = [-5.0, -5.0] # start
    Xe = [7.0, -13.0] # end
    v = Xe - Xi # vector from start to end
    v ./= norm(v)
    n = nullspace(adjoint(v)) # vector normal to v
    # get a covariance matrix with more covariance along `n` than `v`
    θ = atan(n[2], n[1])
    R = [cos(θ) -sin(θ);sin(θ) cos(θ)] #Rotation matrix
    w1² = w1*w1
    Σ = fill(0.0, 2,2)
    Σ[1,1] = w1²
    Σ[2,2] = 1.5
    Σ = R*Σ*R'
    f1,g1 = get_potential(Σ, Xi,A0)

    # basin
    wf² = wf*wf
    Σb = fill(0.0, 2,2)
    Σb[1,1] = wf²
    Σb[2,2] = 1.5*wf²/w1²
    Σb = R*Σb*R'
    fb,gb = get_potential(Σb, Xi, b)
    # the end state 
    w2²= w2*w2
    Σe = [w2² 0.0; 0.0 w2²]
    f2,g2 = get_potential(Σe, Xe, -A0)
    (X,bb=b)->f1(X)+fb(X,bb)+f2(X), (X,bb=b)->g1(X)+gb(X,bb)+g2(X)
end


function get_attractors2(;w1=sqrt(5.0), w2=sqrt(7.0), wf=sqrt(3.5), A0=7.0, b=-4.0, A2=7.0, zmin=-3.2,ϵ2=2.0, ϵ1=ϵ2)
    Xi = [-5.0, -5.0] # start
    Xe = [7.0, -13.0] # end
    v = Xe - Xi # vector from start to end
    v ./= norm(v)
    n = nullspace(adjoint(v)) # vector normal to v
    # get a covariance matrix with more covariance along `n` than `v`
    θ = atan(n[2], n[1])
    R = [cos(θ) -sin(θ);sin(θ) cos(θ)] #Rotation matrix
    w1² = w1*w1
    Σ = fill(0.0, 2,2)
    Σ[1,1] = w1²
    Σ[2,2] = w1²/ϵ1^2
    Σ = R*Σ*R'

    # basin
    wf² = wf*wf
    Σb = fill(0.0, 2,2)
    Σb[1,1] = wf²
    Σb[2,2] = wf²/ϵ2^2
    Σb = R*Σb*R'
    f1,g1,init_func = get_combined_potential(Σ, Σb, Xi, A0, b, zmin)
    # the end state 
    w2²= w2*w2
    Σe = [w2² 0.0; 0.0 w2²]
    f2,g2 = get_potential(Σe, Xe, -A2)
    (X,bb=b,s=1.0,bb2=-A2,ss=1.0)->f1(X,bb,s,ss)+f2(X,bb2), (X,bb=b,s=1.0,bb2=-A2,ss=1.0)->g1(X,bb,s,ss)+g2(X,bb2), init_func
end

function flat_bottom(Λ::Vector{Float64}, b::Float64, b0=0.8*b)
    Σ = diagm(Λ[1:2])
    Σi = inv(Σ)
    c = -2*log(b0/b) 
    f1(X) = (d=X; b*exp(-d'*Σi*d/2)) 
    f2(X) = (d=X;d2=d'*Σi*d; d2 < c ? -sqrt(Λ[3]*(c - d[1]^2/Λ[1] - d[2]^2/Λ[2])) : 0.0)
    f3(X) = f1(X) + f2(X)
end

function ellipsoid(a::Float64, b::Float64, c::Float64,r::Float64)
    f1(X) = (d=X'*X; c*sqrt(r-X[1]^2/a - X[2]^2/b))
end



function get_potential(Σ::Matrix{Float64}, Xi::Vector{Float64},A0::Float64)
    Σi = inv(Σ) 
    f1 = (X,b=A0)->(d = Xi-X; b*exp(-d'*Σi*d/2.0))
    # gradient at X
    g1 = (X,b=A0)->(d=Xi-X; -f1(X,b)/2*(Σi + Σi')*d)
    f1,g1
end

function get_flat_bottom_potential(Σ::Matrix{Float64}, Xi::Vector{Float64}, A0::Float64, zmin=-Inf)
    Σi = inv(Σ)
    f1 = (X,b=A0)->(d = Xi-X; max(b*exp(-d'*Σi*d/2), zmin))
    g1 = (X,b=A0)->(d = Xi-X; z = f1(X,b); z < zmin ? [0.0, 0.0] : -z/2*(Σi + Σi')*d)
    f1,g1
end

function get_combined_potential(Σ1::Matrix{Float64}, Σ2::Matrix{Float64}, Xi::Vector{Float64}, A1::Float64, A2::Float64, zmin=-Inf)
    Σ1i = inv(Σ1)
    Σ2i = inv(Σ2)
    ee = eigen(Σ2i)
    Λ = ee.values 
    P = ee.vectors
    #find a point on the ellipse where the value of 
    y2 = 0.0
    c = -2*log(zmin/A2)
    init_func(q) = random_ellipse(Σ2,q*sqrt(c))
    function f1(X, b=A2, width_scale=1.0,s=1.0)
        _Σ2i = Σ2i./width_scale
        d = Xi-X
        dp1 = d'*Σ1i*d
        z1 = A1*exp(-dp1/2.0)
        z2 = b*exp(-d'*_Σ2i*d/2.0)
        #@info "c" zmin b 
        if b < zmin
            # find dᵗΣ⁻¹d corresponding to the value zmin; we won't allow the potential to dip below zmin here 
            c = -2*log(zmin/b)
            # we just need a single point on the ellipse, so set y₁=0 and solve for y₂.
            y1 = sqrt(width_scale*c/ee.values[1])
            # we solved y₂ in a system where the inverse covariance matrix was diagnoal, so transform back
            Xp = ee.vectors'*[y1,y2]
            dp = Xp
            dp2 = dp'Σ1i*dp
            z1min = A1*exp(-dp2/2.0) 
            if dp1 < dp2
                z1 = z1min
                z2 = zmin
            end
        end
        s*(z1 + z2)
    end

    function g1(X,b=A2, width_scale=1.0,s=1.0)
        # we want to increase the width, which means decreasing the inverse
        _Σ2i = Σ2i./width_scale
        d = Xi-X
        dd1 = d'*Σ1i*d
        dd2 = d'*_Σ2i*d
        z1 = A1*exp(-dd1/2.0)
        z2 = b*exp(-dd2/2.0)
        if z2 <= zmin
            return [0.0, 0.0]
        end
        s*(-z1*(Σ1i + Σ1i')*d/2 - z2*(_Σ2i + _Σ2i')*d/2)
    end
    f1, g1, init_func
end

function get_trajectory(func::Function, gfunc::Function, X0::Vector{Float64},nframes=100,σn=0.0, dt=1.0;
                                                                    bump_amp=0.01, bump_time=20, bump_dur=2)
    b0 = 4.0
    nd = length(X0)
    X = fill(0.0,nd, nframes)
    X[:,1] = X0
    for i in 2:nframes
        if bump_time <= i < bump_time + bump_dur
            j = i - bump_time
            if j < bump_dur/2
                b = b0 - 2*(b0 - bump_amp)*j/bump_dur
            else
                b = bump_amp - 2*(bump_amp-b0)*(j-div(bump_dur,2))/bump_dur
            end
        else
            b = b0
        end
        ΔX = gfunc(X[:,i-1],b)*dt
        X[:,i] = X[:,i-1] + ΔX
    end
    X
end

function random_ellipse(a,b, c)
    x = -c*b .+ 2*c*b*rand()
    ym = sqrt.(a*a*(c*c .- x.*x./b^2))
    y = -ym .+ 2*ym.*rand()
    x,y
end

function random_ellipse(Σ, c)
    ee = eigen(Σ)
    x,y = random_ellipse(1.0/sqrt(ee.values[1]),1.0/sqrt(ee.values[2]),c)
    ee.vectors*[x,y]
end

function animate_manifold(func::Function, gfunc::Function;nframes=100,σn=0.0, dt=1.0,
                                           bump_amp=1.5, max_width_scale=2, bump_time=20, bump_dur=2,well_min=0.0,basin_scale_min=1.0,
                                           b0=5.5,w0=1.0, b20=0.01,b1=7.0, rebound=true, ntrials=1, freeze_before_bump=false,ifunc=(c)->c*randn(2),r0=2.0)

    xx =  -10:0.1:15.0
    yy = -25:0.1:0.0 
    bs = Observable(1.0)
    b = Observable(-b0)
    b2 = Observable(-b20)
    w = Observable(w0)
    zmin = -1.1*well_min
    X0 = Observable([[-5.0, -5.0] + ifunc(r0) for i in 1:ntrials])

    zz = lift(b,w,b2,bs) do _b, _w,_b2,_bs
        func.([[x,y] for x in xx, y in yy],_b, _w,_b2,_bs)
    end
    X = lift(b,X0,w,b2,bs) do _b,_X0,_w, _b2, _bs
        [Makie.Point3f0(_X[1], _X[2], func(_X,_b,_w,_b2,_bs)) for _X in _X0]
    end
    Xl = Observable(cat([[Point3f0(_X0[1], _X0[2], zmin),Point3f0(NaN,NaN,NaN)] for _X0 in X0[]]...,dims=1))
    pidx = [2:2:2*ntrials;]
    on(X0) do _X0
        _Xl = Xl[]
        for (ii,_x0) in enumerate(_X0)
            insert!(_Xl,  pidx[ii], Makie.Point3f0(_x0[1], _x0[2], zmin))
            pidx[ii:end] .+= 1
        end
        Xl[] = _Xl
    end

    fig = Figure(resoltion=(500,500))
    ax = Axis3(fig[1,1])
    zlims!(ax, zmin, maximum(zz[]))
    ax.title = "Frame 1/$nframes"
    surface!(ax, xx, yy, zz)
    meshscatter!(ax, X,markersize=0.25)
    lines!(ax, Xl,color="black")
    lpoints = decompose(Point3f0, Circle(Point2f0(7.0,-13.0), 0.1*sqrt(60/2)))
    lpoints .+= Point3f0(0.0, 0.0, zmin)
    lines!(ax, lpoints)
    @async for i in 2:nframes
        if bump_time <= i < bump_time + bump_dur
            #TODO Also increase width here
            j = i - bump_time
            if rebound
                half_time = bump_dur/2
            else
                half_time = bump_dur
            end
            if j < half_time 
                b[] = -(b0 - (b0 - bump_amp)*j/half_time)
                b2[] = -(b20 - (b20 - well_min)*j/half_time)
                w[] = w0 - (w0 - max_width_scale)*j/half_time
                bs[] = max(1.0 - (1.0 - basin_scale_min)*j/half_time, 0.0)
            else
                b[] = -(bump_amp - (bump_amp-b0)*(j-half_time)/bump_dur)
                b2[] = -(well_min - (well_min-b20)*(j-half_time)/bump_dur)
                # can this become negative
                w[] = max_width_scale - (max_width_scale-w0)*(j-half_time)/bump_dur
                bs[] = max(basin_scale_min - (basin_scale_min-1.0)*(j-half_time)/bump_dur, 0.0)
            end
        else
            #b[] = -b0
            #w[] = w0
        end
        _X0 = X0[]
        if !freeze_before_bump || i >= bump_time+bump_dur
            for _x0 in _X0
                ΔX = gfunc(_x0, b[],w[],b2[],bs[])
                _x0 .= _x0 + ΔX*dt + σn*randn(2)
            end
            X0[] = _X0
        end
        yield()
        sleep(0.1)
        ww = round(w[],sigdigits=2)
        bb = round(b[], sigdigits=2)
        ax.title = "Frame $i/$nframes width=$(ww) height=$(bb)"
    end
    fig
end

function run_model(;redo=false, do_save=true,σ²0=1.0,τ=3.0,σ²n=0.0, nd=78,n_init_points=1,
                                               curve_data_file="curved_new_long_data.jld2",
                                               idx0=30)
    @assert σ²0 >= σ²n
    h = UInt32(0)
    h = crc32c(string(σ²0),h)
    h = crc32c(string(τ),h)
    h = crc32c(string(σ²n),h)
    if nd != 7
        h = crc32c(string(nd),h)
    end
    if n_init_points != 1
        h = crc32c(string(n_init_points),h)
    end
    if curve_data_file != "curved_new_long_data.jld2"
        h = crc32c(curve_data_file,h)
    end
    if idx0 != 30
        h = crc32c(string(idx0),h)
    end
    q = string(h, base=16)
    fname = "model_full_space_results_$q.jld2"
    if isfile(fname) && !redo
        qq = JLD2.load(fname)
        results = NamedTuple(Symbol.(keys(qq)) .=> values(qq))
    else
        xy2 = [7.0 -13.0] # pre-mov basin
        w2 = 60.0 # pre-mov basin width
        curvex, curvey = JLD2.load(curve_data_file,"curvex","curvey")
        # split into trials
        pidx = findall(isnan, curvex)
        curves = [[curvex[pp0+1:pp1-1] curvey[pp0+1:pp1-1]] for (pp0,pp1) in zip(pidx[1:end-1], pidx[2:end])]
        ntrials = length(curves)
        # compute reaction time
        eeidx = fill(0, ntrials)
        eeidxs = fill(0, ntrials)
        curvesp = Vector{Matrix{Float64}}(undef, ntrials)
        path_length = fill(0.0, ntrials)
        for (ii,curve) in enumerate(curves)
            # to get a finer resolution reaction time, we'll first upsample the trajectories and repeat the above analysis
            spl = ParametricSpline(permutedims(curve[idx0:end,:], [2,1]))
            curvesp[ii] = permutedims(Dierckx.evaluate(spl, range(extrema(spl.t)...; length=10*length(spl.t))),[2,1])
            d = sqrt.(dropdims(sum(abs2, curvesp[ii] .- xy2,dims=2),dims=2))
            _eeidx = findfirst(d .< 0.01*w2)
            if _eeidx != nothing
                eeidx[ii] = _eeidx
                path_length[ii] = sum(sqrt.(sum(abs2, diff(curvesp[ii][1:_eeidx,:],dims=1),dims=2)))
            end
        end
        #rt = log.(eeidx)
        rt = 10*path_length
        tidx = findall(rt .> 0)
        path_length = path_length[tidx]
        ntrials = length(tidx)
        rt = rt[tidx]
        @info "Rt-range" extrema(rt)
        eeidx = eeidx[tidx]
        curvesp = curvesp[tidx]
        max_rt_idx = argmax(rt)
        min_rt_idx = argmin(rt)
        np_min = 10
        # 
        Δt = round(Int64, (eeidx[min_rt_idx]-1)/np_min)
        Δt = 1

        # create a cue-aligned higher dimensional population response
        qq,_ = qr(randn(nd,nd))
        W = qq[:,1:2]
        Y = fill(0.0, size(curvesp[1],1),nd, ntrials)
        Ys = fill!(similar(Y), NaN)
        Σ = diagm(fill(sqrt(σ²0/nd), nd)) 
        Ym = fill(0.0, ntrials, nd)
        Ye = fill!(similar(Ym), 0.0)
        Y0 = fill!(similar(Ym), 0.0)
        pl = fill(0.0, ntrials)
        σ² = fill(0.0, size(curvesp[1],1))
        for i in 1:length(σ²)
            _Σ = cov(cat([curve[i,:] for curve in curvesp]...,dims=2),dims=2)
            u,s,v = svd(_Σ)
            σ²[i] = sum(s) 
        end
        nruns = 50
        pltr = fill(0.0, ntrials, nruns)
        plf = fill(0.0, ntrials, nruns)
        pl = fill(0.0, ntrials)
        r²0 = fill(0.0, nruns)
        pv0 = fill(0.0, nruns)
        r²m = fill(0.0, nruns)
        r²e = fill(0.0, nruns)
        r²pl = fill(0.0, nruns)
        r²plf = fill(0.0, nruns)
        r²hr = fill(0.0, nruns)
        r²cv = fill(0.0, nruns)
        r² = fill(0.0, size(Y,1), nruns)
        r²s = fill(0.0, size(Y,1), nruns)
        σ²f = fill!(similar(r²), 0.0)
        ys = fill(0.0, size(Y,1), size(Y,2))
        # correlated noise
        #Q = [1.0/sqrt(2π*τ^2)*exp(-(t1-t2)^2/(2*τ^2)) for t1 in 1:size(Y,1), t2 in 1:size(Y,1)]
        Q = correlated_noise_process(1.0:size(Y,1), τ,1, (σ²0-σ²n)/nd, σ²n/nd)
        @info "Q" size(Q)
        A,U = cholesky(Q) 
        @info "A" size(A)
        @showprogress for r in 1:nruns
            # maybe we need to do some smoothing here?
            for i in 1:ntrials
                curve = curvesp[i]
                for j in 1:size(Y,1)
                    Y[j,:,i] .= W*curve[j,:] 
                end
                # add correlated noise
                Y[:,:,i] .+= A*randn(size(Q,1),nd)
                #Y[:,:,i] .+= randn(size(Y,1),7)*Σ
                Ym[i,:] = Y[1+div(eeidx[i],2),:,i]
                Ye[i,:] = Y[eeidx[i],:,i]
                Y0[i,:] = dropdims(mean(Y[1:n_init_points,:,i],dims=1),dims=1)
                plf[i,r] = sum(sqrt.(sum(abs2, diff(Y[2:eeidx[i],:,i],dims=1),dims=2)))
                # TODO: Do this on the sub-sampled trajectories
            end
            Δ = plf[min_rt_idx,r]/np_min
            # sub-sample this trajectory using steps of length Δ along the path
            for i in 1:ntrials
                Ys[1:end-1,:,i],idxs = subsample_trajectory(Y[2:end,:,i], Δ)
                j = argmax(idxs)
                pltr[i,r],pidx = MovementDecoders.compute_triangular_path_length2(Ys[1:j, :,i])
            end
            # initial position
            #X0 = permutedims(cat([curve[5,:] for curve in curvesp]...,dims=2),[2,1])
            # To make it as similar to the model as possible, we use factor analysis here
            fa = MultivariateStats.fit(MultivariateStats.FactorAnalysis, permutedims(Y0,[2,1]);maxoutdim=1,method=:em)
            Z0 = permutedims(predict(fa, permutedims(Y0,[2,1])),[2,1])
            β,r²0[r], pv0[r],rss = MovementDecoders.llsq_stats(Z0, rt)
            r²0[r] = adjusted_r²(r²0[r], ntrials, length(β))
            @debug "Regress initial" r²0[r] β
            # mid-point 
            #Xm = permutedims(cat([curve[1+div(_eeidx,2),:] for (curve,_eeidx) in zip(curvesp,eeidx)]...,dims=2), [2,1])
            #midpoints = [CartesianIndex(1+div(_eeidx[i],2),i) for i in 1:length(_eeidx)]
            fa = MultivariateStats.fit(MultivariateStats.FactorAnalysis, permutedims(Ym,[2,1]);maxoutdim=1, method=:em)
            Zm = permutedims(predict(fa, permutedims(Ym,[2,1])),[2,1])
            βm,r²m[r], pvm,rssm = MovementDecoders.llsq_stats(Zm, rt)
            r²m[r] = adjusted_r²(r²m[r], ntrials, length(βm))
            @debug "Regress mid" r²m[r] βm
            # end-point
            #Xe = permutedims(cat([curve[_eeidx,2,:] for (curve,_eeidx) in zip(curvesp,eeidx)]...,dims=2), [2,1])
            βe,r²e[r], pve,rsse = MovementDecoders.llsq_stats(Ye, rt)
            r²e[r] = adjusted_r²(r²e[r], ntrials, length(βe))
            @debug "Regress end" r²e[r] βe
            βpl,r²pl[r], pvpl,rsspl = MovementDecoders.llsq_stats(pltr[:,r:r], rt)
            r²pl[r] = adjusted_r²(r²pl[r], ntrials, length(βpl))
            @debug "Regress path length" r²pl[r]
            βplf,r²plf[r], pvplf,rssplf = MovementDecoders.llsq_stats(plf[:,r:r], rt)
            r²plf[r] = adjusted_r²(r²plf[r], ntrials, length(βplf))
            @debug "Regress path length" r²plf[r]
            # hierarhical
            βhr,r²hr[r], pvhr,rsshr = MovementDecoders.llsq_stats([Y0 pltr[:,r]], rt)
            r²hr[r] = adjusted_r²(r²hr[r], ntrials, length(βhr))
            @debug "Regress path length + initial" r²hr[r]
            # compute variance as function of time
            #cross-validated to make sure that the path lengths do carry information
            tidx = shuffle(1:ntrials)[1:div(ntrials,2)]
            teidx = setdiff(1:ntrials,tidx)
            βcv,r²cv[r], pvcv,rsscv = MovementDecoders.llsq_stats(repeat(path_length[tidx],1,1), rt[tidx])
            rtp = βcv[1].*path_length[teidx] .+ βcv[end]
            r²cv[r] = 1.0 - sum(abs2, rtp - rt[teidx])/sum(abs2, rt[teidx] .- mean(rt[teidx]))
            r²cv[r] = adjusted_r²(r²cv[r], ntrials, length(βcv))
            rts = shuffle(rt)
            for i in 1:length(σ²)
                _Σ = cov(Y[i,:,:],dims=2)
                u,s,v = svd(_Σ)
                σ²f[i,r] = sum(s)
                β, r²[i,r], _, _ = MovementDecoders.llsq_stats(permutedims(Y[i,:,:],[2,1]),rt) 
                r²[i,r] = adjusted_r²(r²[i,r], ntrials, length(β))
                β, r²s[i,r], _, _ = MovementDecoders.llsq_stats(permutedims(Y[i,:,:],[2,1]), rts)
                r²s[i,r] = adjusted_r²(r²s[i,r], ntrials, length(β))
            end
        end
        results = (r²e=r²e, r²0=r²0, pv0=pv0, r²pl=r²pl, r²hr=r²hr, r²m=r²m, r²=r², r²s=r²s,
                   rt=rt, pltr=pltr, σ²f=σ²f, curves=curvesp,σ²0=σ²0, σ²=σ²,Y=Y, 
                   path_length=path_length, Ys=Ys, plf=plf, r²plf=r²plf, r²cv=r²cv)
        @info "r²plf" r²plf
        if do_save
            JLD2.save(fname, Dict(String(k)=>results[k] for k in keys(results)))
        end
    end
    results
end

"""
````
function subsample_trajectory(Y::Matrx{Float64}, Δ::Float64)
````
Sub-sample the trajectory `Y` using steps `Δ` along the trajectory
"""
function subsample_trajectory(Y::Matrix{Float64}, Δ::Float64)
    Ys = fill!(similar(Y),NaN)
    ks = fill(0, 1:size(Y,1))
    ks[1] = 1
    Ys[1,:] .= Y[1,:]
    j = 1
    k = 1
    while j < size(Y,1)
        d = 0.0
        while k < size(Y,1) && d < Δ
            d += sqrt(sum(abs2, Y[k,:] - Ys[j,:]))
            k += 1
        end
        j += 1
        Ys[j,:] = Y[k,:] 
        ks[j] = k
        if k == size(Y,1)
            break
        end 
    end
    Ys,ks 
end

correlated_noise_process(t::AbstractVector{Float64}, τ::Float64,p::Int64,σ²f=1.0, σ²n=0.0) = correlated_noise_process(t, fill(τ,p),σ²f, σ²n)

function correlated_noise_process(t::AbstractVector{Float64}, τ::Vector{Float64},σ²f=1.0, σ²n=0.0)
    p = length(τ)
    T = length(t)
    K = fill(0.0, p*T, p*T)
    for (i1,t1) in enumerate(t)
        for (i2,t2) in enumerate(t)
            Δt = t1-t2
            Δt2 = Δt*Δt
            q = 0.0
            if t1==t2
                q = σ²n
            end
            for j in 1:p
                K[(i2-1)*p+j,(i1-1)*p+j] = σ²f*exp(-Δt2/(2*τ[j]*τ[j])) + q
            end
        end
    end
    Symmetric(K)
end

function test()
    K1 = MovementDecoders.correlated_noise_process(1:1750.0, 0.5,7)
    A,L = cholesky(K1)
    XX = reshape(L.L*randn(size(K,1)), 7, size(results.Y,1))
    K2 = MovementDecoders.correlated_noise_process(1:1750.0, 0.5,1)
    A,L = cholesky(K2)
    X = A*randn(size(k2,1),7)
    σ²1 = fill(0.0, size(K2,1))
    σ²2 = fill(0.0, size(K2,1))
    for i in 1:length(σ²1)
       # Σ = cov(XX[:,i],)
    end
end
end
