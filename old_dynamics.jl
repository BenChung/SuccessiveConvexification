module OldDynamics
using DifferentialEquations
using ForwardDiff 
using DiffResults
using StaticArrays
using ..ProbInfo, ..LinPoint

const mass_idx = 1
const r_idx = SVector(2,3,4)
const v_idx = SVector(5,6,7)
const qbi_idx = SVector(8,9,10,11)
const omb_idx = SVector(12,13,14)
const state_dim = 14
const control_dim = 3

struct IntegratorParameters{T}
    dt :: Float64
    uk :: SArray{Tuple{3}, T, 1, 3}
    up :: SArray{Tuple{3}, T, 1, 3}
end


struct LinRes
    Ak::Array{Float64,2}
    Bk::Array{Float64,2}
    Ck::Array{Float64,2}
    Sigk::Array{Float64,1}
    Zk::Array{Float64,1}
    iter_end::Array{Float64, 1} #dunno
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

function dx(state::SArray{Tuple{14}, T, 1, 14} where T, u::MArray{Tuple{3}, T, 1, 3} where T, info::ProbInfo)
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

function compute_A(state::SArray{Tuple{14}, T, 1, 14} where T, u::MArray{Tuple{3}, T, 1, 3} where T, target::MArray{Tuple{14,14}, T, 2, 196} where T, info::ProbInfo)
    ForwardDiff.jacobian!(target, istate -> dx(istate, u, info), state)
end

function compute_B(state::SArray{Tuple{14}, T, 1, 14} where T, u::MArray{Tuple{3}, T, 1, 3} where T, target::MArray{Tuple{14,3}, T, 2, 42} where T, info::ProbInfo)
    ForwardDiff.jacobian!(target, iu -> dx(state, iu, info), u)
end

function linearize_dynamics(states::Array{LinPoint,1}, sigma_lin::Float64, dt::Float64, info::ProbInfo)
    #preallocate memory
    const Aval = @MArray zeros(state_dim, state_dim)
    const Bval = @MArray zeros(state_dim, control_dim)
    const clin = @MVector zeros(control_dim)
    const cneg = @MVector zeros(control_dim)
    const cpls = @MVector zeros(control_dim)

    #compute indices into the combined accumulator matrix
    const state_idx = 1
    const A_start = state_idx + 1
    const A_end = A_start + state_dim - 1
    const B_start = A_end + 1
    const B_end = B_start + control_dim - 1
    const C_start = B_end + 1
    const C_end = C_start + control_dim - 1
    const Sig_idx = C_end + 1
    const Z_idx = Sig_idx + 1

    #integrand worker
    #the accumulator is laid out as 
    # state:1 |     Ak:state_dim     | Bkm:control_dim | Bkp:control_dim | sigK:1 | zk:1
    # all are state_dim high. Dims are thus state_dim+control_dim*2+3 by state_dim.
    function integrand(du,
                       u,#::MArray{Tuple{acc_height,acc_width}, Float64, 2, acc_height*acc_width},
                       p::IntegratorParameters,
                       t::Float64)
        lkm = (p.dt-t)/p.dt
        lkp = t/p.dt
        broadcast!(*, cneg, lkm, p.uk)
        broadcast!(*, cpls, lkp, p.up)
        broadcast!(+, clin, cneg, cpls)

        sv = SVector{14}(u[:,state_idx])
        du[:,1] = dx(sv, clin, info)
        compute_A(sv, clin, Aval, info)
        compute_B(sv, clin, Bval, info)
        broadcast!(*, Aval, sigma_lin, Aval)
        broadcast!(*, Bval, sigma_lin, Bval)
        phi = u[:,A_start:A_end]
        iPhi = inv(phi)

        du[:,A_start:A_end] = Aval*phi
        du[:,B_start:B_end] = lkm*(iPhi * Bval)
        du[:,C_start:C_end] = lkp*(iPhi * Bval)
        du[:,Sig_idx] = iPhi*(du[:,1])
        du[:,Z_idx] = iPhi*(-Aval*u[:,1] - Bval * clin)
    end


    results = Array{LinRes,1}(length(states)-1)
    for i=1:length(states)-1
        uk = states[i].control
        up = states[i+1].control
        int_iv = hcat(states[i].state, eye(Float64, state_dim), zeros(Float64, state_dim, control_dim*2 + 2))
        prob = ODEProblem(integrand, int_iv, (0.0,dt), IntegratorParameters(dt, uk, up))
        sol = DifferentialEquations.solve(prob, DP5(); dtmin=0.001, force_dtmin=true)
        result = sol.u[end]

        Ak::Array{Float64,2} = result[:,A_start:A_end]
        Bkm::Array{Float64,2} = Ak*result[:,B_start:B_end]
        Bkp::Array{Float64,2} = Ak*result[:,C_start:C_end]
        Sigk::Array{Float64,1} = Ak*result[:,Sig_idx]
        zk::Array{Float64,1} = Ak*result[:,Z_idx]
        results[i] = LinRes(Ak, 
            Bkm, 
            Bkp, 
            Sigk, 
            zk, 
            result[:,1])
    end
    return results
end

function next_step(dynam, ab, abn, state, control_k, control_kp, sigma, sigHat, relax)
    dk = dynam
    return dk.Ak * state + dk.Bk * control_k + dk.Ck * control_kp + dk.Sigk*sigma + dk.Zk + relax
end

export linearize_dynamics, LinRes
end