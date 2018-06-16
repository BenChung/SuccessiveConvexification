module Dynamics
    using DifferentialEquations
    using ForwardDiff 
    using DiffResults
    using StaticArrays
    using MathOptInterface
    using RocketlandDefns
    
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

    @inline function Omega{T}(omegab::SArray{Tuple{3}, T, 1, 3})
        z = zero(T)
        return @SMatrix [z        -omegab[1] -omegab[2] -omegab[3];
                        omegab[1] z           omegab[3] -omegab[2];
                        omegab[2] -omegab[3]  z          omegab[1];
                        omegab[3]  omegab[2] -omegab[1]  z         ]
    end

    @inline function dx(output, state::StaticArrays.SArray{Tuple{14},T,1,14} where T, u::SArray{Tuple{3}, T, 1, 3} where T, mult, info::ProbInfo)
        qbi = state[qbi_idx]
        omb = state[omb_idx]
        #=
        #bodyaero_lut = (b.v, v.v) => (body_axis, velocity_axis)
        body_up = DCM(qbi) * (SVector(1,0,0))
        mach_v = state[5:7] ./ speed_of_sound
        body_c, vel_c = bodyaero_lut(dot(body_up, mach_v), dot(mach_v, mach_v))
        aero_acc = ((mach_v * vel_c + body_c * body_up)*speed_of_sound)/state[mass_idx]

        #control_lut = (mach) => (lift_max, drag_max)
        Lmax, Dmax = control_lut(mach)
        aero_control = SVector{3}(0.5*Dmax*(2 + u[4]^2 + u[5]^2), Lmax*u[4],Lmax*u[5])
        =#
        thr_acc = DCM(qbi) * (u[thr_idx]/state[mass_idx] #= + aero_control =#)
        acc = thr_acc # + aero_acc
        rot_vel = 0.5*Omega(omb)*qbi
        rot_acc = info.jBi*(cross(info.rTB,u) #= + cross(info.rFB,aero_control) =# - cross(omb,info.jB*omb))
        output[:] = (@SVector [-info.a*norm(u), 
                     state[5],state[6],state[7], 
                     acc[1]-info.g0, acc[2], acc[3], 
                     rot_vel[1], rot_vel[2], rot_vel[3], rot_vel[4], 
                     rot_acc[1], rot_acc[2], rot_acc[3]])*mult
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
        return LinRes(SVector{14}(x),SMatrix{14,21}(hcat(dp...)) + hcat(eye(14), zeros(14,7)))
    end

    function linearize_dynamics(states::Array{LinPoint,1}, sigma_lin::Float64, dt::Float64, info::ProbInfo)
        results = Array{LinRes,1}(length(states)-1)
        fun = make_dyn(dt,info)

        cstate = states[1]
        nstate = states[2]
        ip = vcat(zeros(14), cstate.control, nstate.control, sigma_lin)
        prob = ODELocalSensitivityProblem(fun, vcat(cstate.state, Float64[]), (0.0,dt), ip)
        integrator = init(prob, Vern9(),abstol=1e-14,reltol=1e-14, save_everystep=false)
        for i=1:length(states)-1
            cstate = states[i]
            nstate = states[i+1]
            ip[:] = SVector{21}(vcat(zeros(14), cstate.control, nstate.control, sigma_lin))
            reinit!(integrator, vcat(cstate.state, zeros(294)), t0=0.0, tf=dt)
            results[i] = linearize_segment(integrator,dt)
        end
        return results
    end
    export linearize_dynamics, next_step
end