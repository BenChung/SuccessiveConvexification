using ForwardDiff
using DiffResults
using DifferentialEquations
using LinearAlgebra

struct ProbInfo
    a::Float64
    g0::Float64
    sos::Float64
    jB::Matrix{Float64}
    jBi::Matrix{Float64}
    rTB::Vector{Float64}
end

const mass_idx = 1
const r_idx = 2:4 # SVector(2,3,4)
const v_idx = 5:7 # SVector(5,6,7)
const qbi_idx = 8:11 # SVector(8,9,10,11)
const omb_idx = 12:14 # SVector(12,13,14)

const thr_idx = 1:3# SVector(1,2,3)
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

@inline function dx_internal(outp, state::AbstractArray{T} where T, u::AbstractArray{T} where T, mult, info::ProbInfo)
    qbi = state[qbi_idx]
    omb = state[omb_idx]

    thr_acc = DCM(qbi) * u[thr_idx]
    acc = thr_acc./state[mass_idx]
    rot_vel = 0.5*Omega(omb)*qbi
    rot_acc = info.jBi*(cross(info.rTB,u) - cross(omb,info.jB*omb))
    outp .= [-info.a*sqrt(sum(u .^ 2)), 
                 state[5],state[6],state[7], 
                 acc[1]-info.g0, acc[2], acc[3], 
                 rot_vel[1], rot_vel[2], rot_vel[3], rot_vel[4], 
                 rot_acc[1], rot_acc[2], rot_acc[3]] .* mult

end

function dx(du,u,p,t)
	et = p[1]
	c1 = p[2]
	c2 = p[3]
	sig = p[4]
	dx_internal(du, u, t/et * c1 + (et - t)/et * c2, sig, p[5])
end

ip = ProbInfo(0.01, 1.0, 5.0, [0.01 0.0 0.0; 0.0 0.01 0.0; 0.0 0.0 0.01], [100.0 0.0 0.0; 0.0 100.0 0.0; 0.0 0.0 100.0], [-0.01, 0.0, 0.0])

function simulate(outp, inp, et, tol)
    prob = ODEProblem(dx, inp[1:14], convert.(eltype(inp), (0.0,et)), [et, inp[15:17], inp[18:20], inp[21], ip])
    integrator = init(prob, BS3(), abstol=tol, reltol=tol, save_everystep=false)
    sol = solve!(integrator)
    outp .= sol[end]
end

function sensitivity(inp, et, tol) 
    outp = zeros(14)
    res = DiffResults.JacobianResult(zeros(14), zeros(21))
    ForwardDiff.jacobian!(res, (y,x) -> begin simulate(y, x, et, tol) end, outp, inp)
    return DiffResults.jacobian(res)
end
