module AdDynamics
    using DifferentialEquations
    using ForwardDiff 
    using DiffResults
    using StaticArrays
    using ..ProbInfo, ..LinPoint
    struct LinRes
        endpoint::SArray{Tuple{14},Float64,1,14}
        derivative::SArray{Tuple{14,21},Float64,2,294}
    end
    
    const mass_idx = 1
    const r_idx = SVector(2,3,4)
    const v_idx = SVector(5,6,7)
    const qbi_idx = SVector(8,9,10,11)
    const omb_idx = SVector(12,13,14)

    struct IntegratorParameters{T}
        dt :: Float64
        uk :: SArray{Tuple{3}, T, 1, 3}
        up :: SArray{Tuple{3}, T, 1, 3}
        sigma :: T
        pinfo :: ProbInfo
    end

    function DCM(quat::SArray{Tuple{4}, T, 1, 4} where T)
        q0 = quat[1]
        q1 = quat[2]
        q2 = quat[3]
        q3 = quat[4]

        return transpose(@SMatrix [(1-2*(q2^2 + q3^2))         (2*(q1 * q2 + q0 * q3))  (2*(q1*q3 - q0 * q2));
                        (2*(q1 * q2 - q0 * q3))     (1-2*(q1^2 + q3^2))      (2*(q2*q3 + q0 * q1));
                        (2*(q1 * q3 + q0 * q2))     (2*(q2*q3 - q0 * q1))    (1-2*(q1^2 + q2^2)) ])
    end

    function Omega{T}(omegab::SArray{Tuple{3}, T, 1, 3})
        z = zero(T)
        return @SMatrix [z        -omegab[1] -omegab[2] -omegab[3];
                        omegab[1] z           omegab[3] -omegab[2];
                        omegab[2] -omegab[3]  z          omegab[1];
                        omegab[3]  omegab[2] -omegab[1]  z         ]
    end

    function dx(state::SArray{Tuple{14}, T, 1, 14} where T, u::SArray{Tuple{3}, T, 1, 3} where T, info::ProbInfo)
        qbi = state[qbi_idx]
        omb = state[omb_idx]
        thr_acc = DCM(qbi) * u/state[mass_idx]
        grv_acc = @SVector [-info.g0,0,0]
        acc = thr_acc + grv_acc
        rot_vel = 0.5*Omega(omb)*qbi
        rot_acc = info.jBi*(cross(info.rTB,u) - cross(omb,info.jB*omb))
        return @SVector [-info.a*norm(u), 
                state[5],state[6],state[7], 
                acc[1], acc[2], acc[3], 
                rot_vel[1], rot_vel[2], rot_vel[3], rot_vel[4], 
                rot_acc[1], rot_acc[2], rot_acc[3]]
    end

    function dynamics_sim(state::SArray{Tuple{14}, T, 1, 14} where T, p::IntegratorParameters, t)
        lkm = (p.dt-t)/p.dt
        lkp = t/p.dt
        u = p.uk*lkm + p.up*lkp
        p.sigma*dx(state,u,p.pinfo)
    end

    macro genIdx(start,last)
        return :(@SVector [i for i=$start:$last])
    end
    const stateC = @genIdx(1,14)
    const ukC = @genIdx(15,17)
    const upC = @genIdx(18,20)
    const sigmaC = 21
    function fdiff(pinfo,dt)
        function f(p)
            state = p[stateC]
            uk = p[ukC]
            up = p[upC]
            sigma = p[sigmaC]
            prob = ODEProblem(dynamics_sim, state, eltype(p).((0.0,dt)), IntegratorParameters(dt,uk,up,sigma,pinfo))
            DifferentialEquations.solve(prob, Tsit5(),reltol=1e-8,abstol=1e-8,force_dtmin=true,dtmin=0.0001,save_everystep=false,saveat=[dt])[end]
        end
        return f
    end

    function linearize_segment(fun, sigma_lin::Float64, istate::SVector{14}, c1::SVector{3}, c2::SVector{3})
        ip = SVector{21}(vcat(istate, c1, c2, sigma_lin))
        res = DiffResults.JacobianResult((@MVector zeros(14)),MVector(ip))
        ForwardDiff.jacobian!(res, fun, ip)
        jcb = DiffResults.jacobian(res)
        return LinRes(DiffResults.value(res),SMatrix{14,21}(jcb))
    end

    function linearize_dynamics(states::Array{LinPoint,1}, sigma_lin::Float64, dt::Float64, info::ProbInfo)
        results = Array{LinRes,1}(length(states)-1)
        fun = fdiff(info,dt)
        for i=1:length(states)-1
            cstate = states[i]
            nstate = states[i+1]
            results[i] = linearize_segment(fun, sigma_lin, cstate.state, cstate.control, nstate.control)
        end
        return results
    end
    function next_step(dynam, ab, abn, state, control_k, control_kp, sigma, sigHat, relax)
        ctrl = vcat(state-ab.state, control_k-ab.control, control_kp-abn.control,sigma-sigHat)
        return dynam.derivative *ctrl + dynam.endpoint + relax
    end
    export linearize_dynamics, next_step, LinRes
end