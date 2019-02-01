module Dynamics
    using DifferentialEquations
    using ForwardDiff 
    using DiffResults
    using LinearAlgebra
    import StaticArrays
    import SymEngine
    import ..SymbolicUtils
    using ..RocketlandDefns
    using ..Aerodynamics
    
    const mass_idx = 1
    const r_idx = 2:4 # SVector(2,3,4)
    const v_idx = 5:7 # SVector(5,6,7)
    const qbi_idx = 8:11 # SVector(8,9,10,11)
    const omb_idx = 12:14 # SVector(12,13,14)

    const thr_idx = 1:3# SVector(1,2,3)

    struct IntegratorParameters{T}
        dt :: Float64
        uk :: Matrix{T}
        up :: Matrix{T}
        sigma :: T
        pinfo :: ProbInfo
    end

    @inline function DCM(quat::Vector{T}) where T
        q0 = quat[1]
        q1 = quat[2]
        q2 = quat[3]
        q3 = quat[4]
        p1 = q1 * q2
        p2 = q0 * q3
        p3 = q1 * q3
        p4 = q0 * q2
        p5 = q2 * q3
        p6 = q0 * q1

        return  T[(1-2*(q2^2 + q3^2)) (2*(p1 - p2))       (2*(p3 + p4));
                        (2*(p1 + p2))        (1-2*(q1^2 + q3^2)) (2*(p5 - p6));
                        (2*(p3 - p4))        (2*(p5 + p6))       (1-2*(q1^2 + q2^2)) ]
    end

    @inline function Omega(omegab::Vector{T}) where T
        z = zero(T)
        return T[z        -omegab[1] -omegab[2] -omegab[3];
                       omegab[1] z           omegab[3] -omegab[2];
                       omegab[2] -omegab[3]  z          omegab[1];
                       omegab[3]  omegab[2] -omegab[1]  z         ]
    end

# original DE.jl time: 0.068
# original rk4 time: 0.031

# naive vects DE.jl: 0.8
# naive vects rk4: 0.15



    @inline function dx_static(state::AbstractArray{T} where T, u::AbstractArray{T} where T, mult, info::ProbInfo)
        qbi = state[qbi_idx]
        omb = state[omb_idx]

        aerf = Aerodynamics.aero_force(info.aero,DCM(qbi) * [1.0,0.0,0.0],(@view state[5:7]),info.sos)

        thr_acc = DCM(qbi) * (u[thr_idx])
        aero_acc = aerf
        acc = (thr_acc + aero_acc)./state[mass_idx] # SymEngine.SymFunction("ifnz").(state[mass_idx], )
        rot_vel = 0.5*Omega(omb)*qbi
        rot_acc = info.jBi*(cross(info.rTB,u) - cross(omb,info.jB*omb))
        return ([-info.a*sqrt(sum(u .^ 2)), 
                     state[5],state[6],state[7], 
                     acc[1]-info.g0, acc[2], acc[3], 
                     rot_vel[1], rot_vel[2], rot_vel[3], rot_vel[4], 
                     rot_acc[1], rot_acc[2], rot_acc[3]]) .* mult
    
    end

    @inline function dx(output, state::AbstractArray{T} where T, u::AbstractArray{T} where T, mult, info::ProbInfo)
        output[:] = dx_static(state, u, mult, info)
        0
    end

    macro genIdx(start,last)
        #return :(@SVector [i for i=$start:$last])
    end
    const stateC = 1:14 # @genIdx(1,14)
    const ukC = 15:17 # @genIdx(15,17)
    const upC = 18:20 # @genIdx(18,20)
    const sigmaC = 21

    function make_dyn(dt,pinfo)
        @inline function dynamics_sim(ipm, state, p, t)
            lkm = (dt-t)/dt
            lkp = t/dt
            dx(ipm, state + p[stateC], p[ukC]*lkm + p[upC]*lkp, p[sigmaC], pinfo)
        end
        return dynamics_sim
    end

    struct StateInfo
        controlK::Vector{Float64}
        controlKp::Vector{Float64}
        sigma::Float64
        dt::Float64
        pinfo::ProbInfo
    end
    
    function control_dx(ipm, state, p, t)
        dt = p.dt
        lkm = (dt-t)/dt
        lkp = t/dt
        dx(ipm, state, p.controlK*lkm + p.controlKp*lkp, p.sigma, p.pinfo)
    end

    function predict_state(initial_state, uk, up, sigma, dt, pinfo)
        prob = ODEProblem(control_dx, vcat(initial_state, Float64[]), (0.0,dt), StateInfo(uk, up, sigma, dt, pinfo))
        sol = solve(prob, Vern9(), save_everystep=false)
        return sol[end]
    end

    function linearize_segment(int,dt)
        sol = DifferentialEquations.solve!(int)
        x,dp = extract_local_sensitivities(int.sol,dt)
        return LinRes(x,hcat(dp...) + hcat(Matrix(1.0I, 14, 14), zeros(14,7)))
    end

    function make_dynamics_module(info::ProbInfo, dt::Float64)
        t=SymEngine.symbols("t")
        dt=SymEngine.symbols("dt")
        lkm = (dt-t)/dt
        lkp = t/dt

        #sig: dx(J, inp, dt, pinfo, t)
        dx_fun = SymbolicUtils.make_simplified(:dx, st -> dx_static(st[stateC], st[ukC]*lkm + st[upC]*lkp, st[sigmaC], info), 21)
        genmod = quote
            module Linearizer
                using LinearAlgebra
                using DifferentialEquations
                using ForwardDiff
                import DiffResults
                using ..RocketlandDefns
                import ..Aerodynamics

                mutable struct IntegratorCache
                    integrator::Any
                end

                function drag(cos_aoa::T,mach::T,p::ProbInfo) where T
                    int_aoa = zero(T)
                    if mach > zero(T)
                        int_aoa = clamp(cos_aoa/(mach*p.sos),-1.0,1.0)
                    end
                    dragf = Aerodynamics.direct_drag(p.aero,int_aoa,mach)::T
                    return dragf
                end
                function lift(cos_aoa::T,mach::T,p::ProbInfo) where T
                    int_aoa = zero(T)
                    if mach > zero(T)
                        int_aoa = clamp(cos_aoa/(mach*p.sos),-1.0,1.0)
                    end
                    return Aerodynamics.direct_lift(p.aero,int_aoa,mach)::T
                end
                $dx_fun

                @inline function ifnz(val::T, nz::V) where {T,V}
                    if !iszero(val)
                        return nz
                    else 
                        return zero(V)
                    end
                end

                @inline function dx_integrator(du,u,p,t)
                    fill!(du, zero(eltype(du)))
                    res = dx(du, u, p[1], p[2], t)
                    return 0.0
                end

                function simulate(outp, inp::Vector{T}, dt::Float64, info::ProbInfo, cache::IntegratorCache) where T
                    state = (@view inp[1:14] )
                    start_ctrl = (@view inp[15:17])
                    ed_ctrl = (@view inp[18:20])
                    i_sigma = inp[21]*dt
                    if isnothing(cache.integrator)
                        prob = ODEProblem(dx_integrator, inp, convert.(eltype(inp), (0.0,dt)), [dt,info])
                        cache.integrator = init(prob, BS3(), abstol=1e-2, reltol=1e-2, save_everystep=false)
                    else
                        reinit!(cache.integrator, inp, 
                            t0 = convert(eltype(inp), 0.0), tf = convert(eltype(inp), dt),
                            reset_dt = false)
                    end

                    sol = solve!(cache.integrator)
                    outp .= sol[end][1:14]
                end

                function allocate_config()
                    cfg = ForwardDiff.JacobianConfig(nothing, zeros(14), zeros(21))
                    fill!(cfg.duals[1], zero(eltype(cfg.duals[1])))
                    fill!(cfg.duals[2], zero(eltype(cfg.duals[2])))
                    return cfg
                end

                function sensitivity(res, inp::Vector{Float64}, dt::Float64, info::ProbInfo, cache::IntegratorCache, cfg) 
                    outp = zeros(14)
                    ForwardDiff.jacobian!(res, (y,x) -> begin simulate(y, x, dt, info, cache) end, outp, inp, cfg)
                end
            end
        end
        Main.eval(genmod.args[2])
    end

    function make_state(a::LinPoint, b::LinPoint, sig::Float64)
        return vcat(a.state, a.control, b.control, sig)
    end

    function linearize_dynamics(states::Array{LinPoint,1}, sigma_lin::Float64, dt::Float64, info::ProbInfo)
        results = Array{LinRes,1}(undef, length(states)-1)
        fun = make_dyn(dt,info)

        cstate = states[1]
        nstate = states[2]
        ip = vcat(zeros(14), cstate.control, nstate.control, sigma_lin)
        prob = ODELocalSensitivityProblem(fun, vcat(cstate.state, Float64[]), (0.0,dt), ip)
        integrator = init(prob, Vern9(),abstol=1e-14,reltol=1e-14, save_everystep=false, force_dtmin = true)

        for i=1:length(states)-1
            cstate = states[i]
            nstate = states[i+1]
            ip[:] .= vcat(zeros(14), cstate.control, nstate.control, sigma_lin)
            reinit!(integrator, vcat(cstate.state, zeros(294)), t0=0.0, tf=dt)
            results[i] = linearize_segment(integrator,dt)
        end
        return results
    end

    @inline function current_control(pc, start_ctrl::AbstractArray{T} where T, ed_ctrl::AbstractArray{T} where T)
        return (1.0-pc) * start_ctrl + pc * ed_ctrl
    end
    function rk4_seg_dyn(dt::Float64, info::ProbInfo)
        function inner(outp::Vector{T}, inp::Vector{T}) where T 
            state = (@view inp[1:14] )
            start_ctrl = (@view inp[15:17])
            ed_ctrl = (@view inp[18:20])
            i_sigma = inp[21]*dt
            npts = 40
            idt = i_sigma/npts
            pcs = 1.0/npts
            ct = 0.0
            pca = 0.0
            for i=0:npts
                ict = current_control(pca, start_ctrl, ed_ctrl)
                mct = current_control(pca + pcs/2, start_ctrl, ed_ctrl)
                ect = current_control(pca + pcs, start_ctrl, ed_ctrl)
                k1 = dx_static(state, ict, idt, info)
                k2 = dx_static(state + k1./2, mct, idt, info)
                k3 = dx_static(state + k2./2, mct, idt, info)
                k4 = dx_static(state + k3, ect, idt, info)
                ct += idt
                pca += pcs
                state += k1./6 + k2./3 + k3./3 + k4./6
            end
            outp[:] = state
        end
        return inner
    end

    function linearize_dynamics_rk4(states::Array{LinPoint,1}, sigma_lin::Float64, dt::Float64, info::ProbInfo)
        results = Array{LinRes,1}(undef, length(states)-1)
        rk4fun = rk4_seg_dyn(dt, info)

        exout = Array{Float64}(undef, 14)
        exin = Array{Float64}(undef, 21)
        res = DiffResults.JacobianResult(exout, exin)
        config = ForwardDiff.JacobianConfig(rk4fun, exout, exin)
        for i=1:length(states)-1
            cstate = states[i]
            nstate = states[i+1]
            rk4ps = Array{Float64}(undef, 14)
            inp = vcat(cstate.state, cstate.control, nstate.control, sigma_lin)
            ForwardDiff.jacobian!(res, rk4fun, rk4ps, inp, config)
            results[i] = LinRes(StaticArrays.SVector{14}(DiffResults.value(res)),StaticArrays.SMatrix{14,21}(DiffResults.jacobian(res)))
        end
        return results
    end

    struct LinearCache
        cfg :: ForwardDiff.JacobianConfig
        cache :: Any # don't know until inner module is generated
        probinfo :: ProbInfo
        base_dt :: Float64
        lin_mod
    end

    function initalize_linearizer(prob::ProbInfo, base_dt::Float64)
        lin_mod = Dynamics.make_dynamics_module(prob, base_dt)
        cache = Base.invokelatest(lin_mod.IntegratorCache, nothing)
        cfg = Base.invokelatest(lin_mod.allocate_config)
        return LinearCache(cfg, cache, prob, base_dt, lin_mod)
    end
    function linearize_dynamics_symb(states::Array{LinPoint,1}, sigma_lin::Float64, cache::LinearCache)
        results = Array{LinRes,1}(undef, length(states)-1)
        exout = Array{Float64}(undef, 14)
        exin = Array{Float64}(undef, 21)
        res = DiffResults.JacobianResult(exout, exin)
        for i=1:length(states)-1
            Base.invokelatest(cache.lin_mod.sensitivity, res, make_state(states[i], states[i+1], sigma_lin),
                cache.base_dt, cache.probinfo, cache.cache, cache.cfg)
            results[i] = LinRes(StaticArrays.SVector{14}(DiffResults.value(res)),
                    StaticArrays.SMatrix{14,21}(DiffResults.jacobian(res)))
        end
        return results
    end
    export linearize_dynamics, next_step
end