module Dynamics
    using DifferentialEquations
    using DiffEqSensitivity
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

    @inline function dx_static(state::AbstractArray{T} where T, u::AbstractArray{T} where T, mult, info::ProbInfo)
        qbi = state[qbi_idx]
        omb = state[omb_idx]

        aerf, bdy_trq = Aerodynamics.aero_force(info.aero,DCM(qbi) * [1.0,0.0,0.0],(@view state[5:7]),info.sos)

        fd1 = cross(DCM(qbi) * [0.0,1.0,0.0], state[5:7])
        fd1 = fd1/sqrt(sum(fd1 .* fd1))
        fd2 = cross(fd1, state[5:7])
        #= ff = u[4] * fd1 + u[5] * fd2 =#

        thr_frc = DCM(qbi) * (u[thr_idx])
        aero_frc = aerf #= + ff =#
        acc = (thr_frc + aero_frc) ./ state[mass_idx]
        rot_vel = 0.5*Omega(omb)*qbi
        aero_trq = #= cross(info.rFB, ff) + =# bdy_trq
        rot_acc = info.jBi*(cross(info.rTB,u[thr_idx]) + aero_trq - cross(omb,info.jB*omb))
        return ([-info.a*sqrt(sum(u[thr_idx] .* u[thr_idx])), 
                     state[5],state[6],state[7], 
                     acc[1]-info.g0, acc[2], acc[3], 
                     rot_vel[1], rot_vel[2], rot_vel[3], rot_vel[4], 
                     rot_acc[1], rot_acc[2], rot_acc[3]]) .* mult
    
    end

    @inline function dx(output, state::AbstractArray{T} where T, u::AbstractArray{T} where T, mult, info::ProbInfo)
        output[:] = dx_static(state, u, mult, info)
        0
    end
    
    @inline function current_control(pc, start_ctrl::AbstractArray{T} where T, ed_ctrl::AbstractArray{T} where T)
        return (1.0-pc) * start_ctrl + pc * ed_ctrl
    end

    function rk4(inp::Vector{T}, dt, info) where T 
        state = (@view inp[1:14] )
        start_ctrl = (@view inp[15:17])
        ed_ctrl = (@view inp[18:20])
        i_sigma = inp[21]*dt
        npts = 1
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
        return state
    end

    const stateC = 1:14
    const ukC = 15:17
    const upC = 18:20
    const sigmaC = 21

    function make_dynamics_module(info::ProbInfo)
        t=SymEngine.symbols("t")
        dt=SymEngine.symbols("dt")
        lkm = (dt-t)/dt
        lkp = t/dt

        #sig: dx(J, inp, dt, pinfo, t)
        dx_fun = SymbolicUtils.make_simplified(:dx, st -> dx_static(st[stateC], st[ukC]*lkm + st[upC]*lkp, st[sigmaC], info), 21)
        ddx_fun = SymbolicUtils.make_jacobian(:ddx, st -> dx_static(st[stateC], st[ukC]*lkm + st[upC]*lkp, st[sigmaC], info), 21)
        #J_fun = SymbolicUtils.make_jacobian(:jac, st -> rk4(st, 1.0, info), 21)
        genmod = quote
            module Linearizer
                using LinearAlgebra
                using ForwardDiff
                import StaticArrays
                using ..RocketlandDefns
                using ..Aerodynamics


                # all of these are called by the generated code
                function clamp_aoa(cos_aoa::T, mach::T,p::ProbInfo) where T
                    int_aoa = zero(T)
                    if mach > zero(T)
                        int_aoa = clamp(cos_aoa/(mach*p.sos),-1.0,1.0)
                    end
                    return int_aoa
                end

                function drag(cos_aoa::T,mach::T,p::ProbInfo) where T
                    int_aoa = clamp_aoa(cos_aoa, mach, p)
                    dragf = direct_drag(p.aero,int_aoa,mach)::T
                    return dragf
                end
                function drag(::Type{Val{:jac}}, cos_aoa::T,mach::T, p::ProbInfo) where T
                    int_aoa = clamp_aoa(cos_aoa, mach, p)
                    return direct_drag(Val{:jac}, p.aero, int_aoa, mach)
                end

                function lift(cos_aoa::T,mach::T,p::ProbInfo) where T
                    int_aoa = clamp_aoa(cos_aoa, mach, p)
                    return direct_lift(p.aero,int_aoa,mach)::T
                end
                function lift(::Type{Val{:jac}}, cos_aoa::T,mach::T,p::ProbInfo) where T
                    int_aoa = clamp_aoa(cos_aoa, mach, p)
                    return direct_lift(Val{:jac}, p.aero, int_aoa, mach)
                end

                function trq(cos_aoa::T,mach::T,p::ProbInfo) where T
                    int_aoa = clamp_aoa(cos_aoa, mach, p)
                    return direct_trq(p.aero,int_aoa,mach)::T
                end
                function trq(::Type{Val{:jac}}, cos_aoa::T,mach::T,p::ProbInfo) where T
                    int_aoa = clamp_aoa(cos_aoa, mach, p)
                    return direct_trq(Val{:jac}, p.aero, int_aoa, mach)
                end

                @inline function ifnz(val::T, nz::V) where {T,V}
                    if !iszero(val)
                        return nz
                    else 
                        return zero(V)
                    end
                end
                @inline function ifnz(::Type{Val{:jac}}, val::T, nz::V) where {T,V}
                    return StaticArrays.SArray{Tuple{2}}(0.0,1.0) # blunt approximation
                end

                $dx_fun
                $ddx_fun
            end
        end
        Main.eval(genmod.args[2])
    end

    struct DxIntegrator{T,V,X} <: Function 
        temp_state::T
        dt::Float64
        info::ProbInfo{V}
        dx::X
    end
    @inline function (ff::DxIntegrator)(du,u,p::Vector{Float64},t::Float64)
        fill!(du, zero(eltype(du)))
        ff.temp_state .= u .+ p
        res = ff.dx(du, ff.temp_state, ff.dt, ff.info, t)
        return nothing
    end

    @inline function ddx_integrator(J,u,p,t, dt, info, ddx)
        fill!(J, zero(eltype(J)))
        for i=1:21
            J[i,i] = 1.0
        end
        return nothing
    end

    struct ParamJac{T,F} <: Function
        temp_state :: Vector{Float64}
        dt::Float64
        info::ProbInfo{T}
        ddx::F
    end
    @inline function (ff::ParamJac)(pJ::Matrix{Float64}, u, p::Vector{Float64}, t::Float64)
        fill!(pJ, zero(eltype(pJ)))
        ff.temp_state .= u .+ p
        res = ff.ddx(pJ, ff.temp_state, ff.dt, ff.info, t)
        return nothing
    end

    mutable struct IntegratorCache
        sim_prob::Any
        sense_prob::Any
        sim_int::Any 
        sense_int::Any
        params::Any
        info::ProbInfo
        function IntegratorCache(prob::DescentProblem, info::ProbInfo, lin_mod)
            return IntegratorCache([prob.mwet; prob.rIi; prob.vIi; prob.qBIi; prob.wBi; 1.0; 0.0; 0.0; 1.0; 0.0; 0.0; 1.0], 1/(prob.K+1), info, lin_mod)
        end
        function IntegratorCache(inp::Vector{Float64}, dt::Float64, info::ProbInfo, lin_mod)
            params = [dt,info]
            dxfun = lin_mod.dx 
            ddxfun = lin_mod.ddx
            dx_int = DxIntegrator(zeros(21), params[1], params[2], dxfun)
            pj_int = ParamJac(zeros(21), params[1], params[2], ddxfun)
            fun = ODEFunction(
                dx_int; 
                jac=(J,u,p,t) -> ddx_integrator(J,u,p,t,params[1],params[2],ddxfun), 
                paramjac=pj_int)
            sim_prob = ODEProblem(fun, 
                zeros(21), convert.(eltype(inp), (0.0,dt)), 
                inp, 
                autodiff=false,
                autojacvec=false)
            sim_int = init(sim_prob, BS3(), abstol=1e-2, reltol=1e-2, save_everystep=false, save_start=false)
            sense_prob = ODEForwardSensitivityProblem(fun, 
                zeros(21), convert.(eltype(inp), (0.0,dt)), 
                inp, 
                ForwardSensitivity(autojacvec=false);
                autodiff=false,
                autojacvec=false)
            sense_int = init(sense_prob, BS3(), abstol=1e-2, reltol=1e-2, save_everystep=false, save_start=false)
            new(sim_prob, sense_prob, sim_int, sense_int, params, info)
        end
    end

    function simulate(res, inp::Vector{Float64}, dt::Float64, cache::IntegratorCache)
        fill!(cache.sim_int.u, 0.0)
        reinit!(cache.sim_int, inp, t0 = convert(eltype(inp), 0.0), tf = convert(eltype(inp), dt), reset_dt=true)
        cache.params[1] = dt 
        solve!(cache.sim_int)
    end

    function sensitivity(res, inp::Vector{Float64}, dt::Float64, cache::IntegratorCache)
        fill!(cache.sense_int.u, 0.0)
        reinit!(cache.sense_int, inp, t0 = convert(eltype(inp), 0.0), tf = convert(eltype(inp), dt), reset_dt=true)
        cache.params[1] = dt 
        solve!(cache.sense_int)
        return extract_local_sensitivities(cache.sense_int.sol, 1, Val(true))
    end

    function predict_state(initial_state, uk, up, sigma, dt, pinfo, cache)
        res = Array{Float64}(undef, 14)
        simulate(res, [initial_state; uk; up; sigma], dt, cache.cache)
        return res
    end

    function linearize_dynamics_symb(states::Array{LinPoint,1}, tf_guess::Float64, base_dt::Float64, cache::IntegratorCache) where {N,K}
        results = Array{LinRes,1}(undef, length(states)-1)
        exout = Array{Float64}(undef, 14)
        exin = Array{Float64}(undef, 25)
        res = DiffResults.JacobianResult(exout, exin)

        for i=1:length(states)-1
            ist = make_state(states[i], states[i+1], tf_guess)
            val, mat = sensitivity(res, ist, base_dt, cache.probinfo, cache)
            results[i] = LinRes(copy(val), copy(mat))
        end
        return results
    end
    export linearize_dynamics, next_step
end