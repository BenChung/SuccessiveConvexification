module Dynamics
    using DifferentialEquations
    using ForwardDiff 
    using DiffResults
    using StaticArrays
    using LinearAlgebra
    using MathOptInterface
    using ..RocketlandDefns
    using ..Aerodynamics
    
    const mass_idx = 1
    const r_idx = SVector(2,3,4)
    const v_idx = SVector(5,6,7)
    const qbi_idx = SVector(8,9,10,11)
    const omb_idx = SVector(12,13,14)

    const thr_idx = SVector(1,2,3)

    struct IntegratorParameters{T}
        dt :: Float64
        uk :: SArray{Tuple{3}, T, 1, 3}
        up :: SArray{Tuple{3}, T, 1, 3}
        sigma :: T
        pinfo :: ProbInfo
    end

    @inline function DCM(quat::SArray{Tuple{4}, T, 1, 4} where T)
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

        return @SMatrix [(1-2*(q2^2 + q3^2)) (2*(p1 - p2))       (2*(p3 + p4));
                        (2*(p1 + p2))        (1-2*(q1^2 + q3^2)) (2*(p5 - p6));
                        (2*(p3 - p4))        (2*(p5 + p6))       (1-2*(q1^2 + q2^2)) ]
    end

    @inline function Omega(omegab::SArray{Tuple{3}, T, 1, 3}) where T
        z = zero(T)
        return @SMatrix [z        -omegab[1] -omegab[2] -omegab[3];
                        omegab[1] z           omegab[3] -omegab[2];
                        omegab[2] -omegab[3]  z          omegab[1];
                        omegab[3]  omegab[2] -omegab[1]  z         ]
    end
    @inline function dx_static(state::StaticArrays.SArray{Tuple{14},T,1,14} where T, u::SArray{Tuple{3}, T, 1, 3} where T, mult, info::ProbInfo)
        qbi = state[qbi_idx]
        omb = state[omb_idx]

        aerf = Aerodynamics.aero_force(info.aero,DCM(qbi) * (SVector(1,0,0)),state[SVector(5,6,7)],info.sos)

        thr_acc = DCM(qbi) * (u[thr_idx]/state[mass_idx] #= + aero_control =#)
        aero_acc = aerf/state[mass_idx]
        acc = thr_acc + aero_acc
        rot_vel = 0.5*Omega(omb)*qbi
        rot_acc = info.jBi*(cross(info.rTB,u) #= + cross(info.rFB,aero_control) =# - cross(omb,info.jB*omb))
        return (@SVector [-info.a*norm(u), 
                     state[5],state[6],state[7], 
                     acc[1]-info.g0, acc[2], acc[3], 
                     rot_vel[1], rot_vel[2], rot_vel[3], rot_vel[4], 
                     rot_acc[1], rot_acc[2], rot_acc[3]])*mult
    
    end

    @inline function dx(output, state::StaticArrays.SArray{Tuple{14},T,1,14} where T, u::SArray{Tuple{3}, T, 1, 3} where T, mult, info::ProbInfo)
        output[:] = dx_static(state, u, mult, info)
        0
    end

    macro genIdx(start,last)
        return :(@SVector [i for i=$start:$last])
    end
    const stateC = @genIdx(1,14)
    const ukC = @genIdx(15,17)
    const upC = @genIdx(18,20)
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
        controlK::SArray{Tuple{3}, Float64, 1, 3}
        controlKp::SArray{Tuple{3}, Float64, 1, 3}
        sigma::Float64
        dt::Float64
        pinfo::ProbInfo
    end
    
    function control_dx(ipm, state, p, t)
        dt = p.dt
        lkm = (dt-t)/dt
        lkp = t/dt
        dx(ipm, SVector{14}(state), p.controlK*lkm + p.controlKp*lkp, p.sigma, p.pinfo)
    end

    function predict_state(initial_state, uk, up, sigma, dt, pinfo)
        prob = ODEProblem(control_dx, vcat(initial_state, Float64[]), (0.0,dt), StateInfo(uk, up, sigma, dt, pinfo))
        sol = solve(prob, Vern9(), save_everystep=false)
        return sol[end]
    end

    function linearize_segment(int,dt)
        sol = DifferentialEquations.solve!(int)
        x,dp = extract_local_sensitivities(int.sol,dt)
        return LinRes(SVector{14}(x),SMatrix{14,21}(hcat(dp...)) + hcat(Matrix(1.0I, 14, 14), zeros(14,7)))
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
            ip[:] = SVector{21}(vcat(zeros(14), cstate.control, nstate.control, sigma_lin))
            reinit!(integrator, vcat(cstate.state, zeros(294)), t0=0.0, tf=dt)
            results[i] = linearize_segment(integrator,dt)
        end
        return results
    end

    @inline function current_control(pc, start_ctrl::SArray{Tuple{3}, T, 1, 3} where T, ed_ctrl::SArray{Tuple{3}, T, 1, 3} where T)
        return (1.0-pc) * start_ctrl + pc * ed_ctrl
    end
    function rk4_seg_dyn(dt::Float64, info::ProbInfo)
        function inner(outp::Vector{T}, inp::Vector{T}) where T 
            k1 = Array{T}(undef, 14)
            k2 = Array{T}(undef, 14)
            k3 = Array{T}(undef, 14)
            k4 = Array{T}(undef, 14)
            state = outp
            state[:] = @view inp[1:14] 
            start_ctrl = SVector{3}(@view inp[15:17])
            ed_ctrl = SVector{3}(@view inp[18:20])
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
                dx_static(k1, SVector{14}(state), ict, idt, info)
                dx_static(k2, SVector{14}(state + k1/2), mct, idt, info)
                dx_static(k3, SVector{14}(state + k2/2), mct, idt, info)
                dx_static(k4, SVector{14}(state + k3), ect, idt, info)
                ct += idt
                pca += pcs
                state[:] += 1/6*k1 + 2/6*k2 + 2/6*k3 + 1/6*k4
            end
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
            results[i] = LinRes(SVector{14}(DiffResults.value(res)),SMatrix{14,21}(DiffResults.jacobian(res)))
        end
        return results
    end
    export linearize_dynamics, next_step
end