using StaticArrays
using LinearAlgebra
using SymEngine

const mass_idx = 1
const r_idx = [2,3,4]
const v_idx = [5,6,7]
const qbi_idx = [8,9,10,11]
const omb_idx = [12,13,14]

struct DescentProblem
    g::Float64
    mdry::Float64
    mwet::Float64
    Tmin::Float64
    Tmax::Float64
    deltaMax::Float64
    thetaMax::Float64
    gammaGs::Float64
    omMax::Float64
    dpMax::Float64 # max dynamic pressure Pa
    jB::Array{Float64,2}
    alpha::Float64
    rho::Float64 # atmospheric density kg/m^3
    sos::Float64 # speed of sound m/s

    rTB::Array{Float64,1}
    rFB::Array{Float64,1}

    rIi::Array{Float64,1}
    rIf::Array{Float64,1}
    vIi::Array{Float64,1}
    vIf::Array{Float64,1}
    qBIi::Array{Float64,1}
    qBIf::Array{Float64,1}
    wBi::Array{Float64,1}
    wBf::Array{Float64,1}

    K::Int64
    imax::Int64
    wNu::Float64
    wID::Float64
    wDS::Float64
    wCst::Float64
    wTviol::Float64
    nuTol::Float64
    delTol::Float64
    tf_guess::Float64

    ri::Float64
    rh0::Float64
    rh1::Float64
    rh2::Float64
    alph::Float64
    bet::Float64

    DescentProblem(;g=1.0,mdry=1.0,mwet=2.0,Tmin=0.3,Tmax=5.0,deltaMax=20.0,thetaMax=90.0,gammaGs=20.0,dpMax=50000,omMax=60.0,
                    jB=diagm(0=>[1e-2,1e-2,1e-2]), alpha=0.01, rho=1.225,rTB=[-1e-2,0,0],rFB=[1e-2,0,0],rIi=[4.0,4.0,0.0],rIf=[0.0,0.0,0.0],
                    vIi=[0,-2,2],vIf=[-0.1,0.0,0.0],qBIi=[1.0,0,0,0],qBIf=[1.0,0,0,0],wBi=[0.0,0.0,0.0],
                    wBf=[0.0,0,0],K=50,imax=15,wNu=1e5,wID=1e-3, wDS=1e-1, wCst=10.0, wTviol=100.0, nuTol=1e-10, delTol = 1e-3, tf_guess=1.0, ri=1.0, rh0=0.0, rh1=0.25, rh2=0.90, alph=2.0, bet=3.2, sos=5.0) =
        new(g,mdry,mwet,Tmin,Tmax,deltaMax,thetaMax,gammaGs,omMax,dpMax,jB,alpha,rho,sos,rTB,rFB,rIi,rIf,vIi,vIf,
            qBIi,qBIf,wBi,wBf,K,imax,wNu,wID,wDS,wCst,wTviol,nuTol,delTol,tf_guess,ri,rh0,rh1,rh2, alph, bet)
end

struct ProbInfo
    a::Float64
    g0::Float64
    sos::Float64
    jB::Matrix{Float64}
    jBi::Matrix{Float64}
    rTB::Vector{Float64}
    ProbInfo(from::DescentProblem) = new(from.alpha, from.g, from.sos, 
        from.jB, inv(from.jB), from.rTB)
end

function DCM(quat::Vector{T} where T)
    q0 = quat[1]
    q1 = quat[2]
    q2 = quat[3]
    q3 = quat[4]

    return transpose([(1.0-2.0*(q2^2.0 + q3^2.0))   (2.0*(q1 * q2 + q0 * q3))  (2.0*(q1*q3 - q0 * q2));
                    (2.0*(q1 * q2 - q0 * q3))     (1.0-2.0*(q1^2.0 + q3^2.0))      (2.0*(q2*q3 + q0 * q1));
                    (2.0*(q1 * q3 + q0 * q2))     (2.0*(q2*q3 - q0 * q1))    (1.0-2.0*(q1^2.0 + q2^2.0)) ])
end

function Omega(omegab::Vector{T}) where T
    z = zero(T)
    return @SMatrix [z        -omegab[1] -omegab[2] -omegab[3];
                    omegab[1] z           omegab[3] -omegab[2];
                    omegab[2] -omegab[3]  z          omegab[1];
                    omegab[3]  omegab[2] -omegab[1]  z         ]
end


function aero_force(bv::Vector{Basic}, 
                    vel::Vector{Basic}, spds::Float64)
    sym_drag_itrp = SymFunction("idrag")
    sym_lift_itrp = SymFunction("ilift")
    pinfo = symbols("pinfo")
    dp = sum(bv .* vel)/sqrt(sum(vel .^ 2.0))
    cos_aoa = dp/sqrt(sum(bv .^ 2.0))
    mach = sqrt(sum(vel .^ 2.0))/spds

    drag = sym_drag_itrp(cos_aoa,mach,pinfo)
    lift = sym_lift_itrp(cos_aoa,mach,pinfo)

    liftd = cross(cross(bv, vel),vel)
    liftd = liftd/sqrt(sum(liftd .^ 2.0))
    dragf = drag*vel/sqrt(sum(vel .^ 2.0))
    liftf = lift*liftd
    return dragf + liftf
end

function dx(state::Vector{T} where T, u::Vector{T} where T, info::ProbInfo)
    qbi = state[qbi_idx]
    omb = state[omb_idx]
    aerf = aero_force(Array(DCM(qbi) * [1.0,0.0,0.0]),state[5:7],info.sos)

    thr_acc = DCM(qbi) * u/state[mass_idx]
    grv_acc = [-info.g0,0.0,0.0]
    aero_acc = aerf/state[mass_idx]

    acc = thr_acc + grv_acc + aero_acc
    rot_vel = 0.5*Omega(omb)*qbi
    rot_acc = info.jBi*(cross(info.rTB,u) - cross(omb,info.jB*omb))
    return [-info.a*sqrt(sum(u .^ 2)), 
            state[5],state[6],state[7], 
            acc[1], acc[2], acc[3], 
            rot_vel[1], rot_vel[2], rot_vel[3], rot_vel[4], 
            rot_acc[1], rot_acc[2], rot_acc[3]]
end