using DifferentialEquations
using Mosek
using MathOptInterface
using MathOptInterfaceMosek

using DifferentialEquations
using DiffResults

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