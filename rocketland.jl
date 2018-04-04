using JuMP
using Mosek
using DifferentialEquations
using ForwardDiff
using Rotations
using StaticArrays

#state indexing
const state_dim = 14
const control_dim = 3
const mass_idx = 1
const r_idx = SVector(2,3,4)
const v_idx = SVector(5,6,7)
const qbi_idx = SVector(8,9,10,11)
const omb_idx = SVector(12,13,14)
const r_idx_it = 2:4
const v_idx_it = 5:7
const qbi_idx_it = 8:11
const omb_idx_it = 12:14
const acc_width = state_dim+control_dim*2+3
const acc_height = state_dim

type DescentProblem
    g::Float64
    mdry::Float64
    mwet::Float64
    Tmin::Float64
    Tmax::Float64
    deltaMax::Float64
    thetaMax::Float64
    gammaGs::Float64
    omMax::Float64
    jB::Array{Float64,2}
    alpha::Float64
    rTB::Array{Float64,1}

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
    nuTol::Float64
    delTol::Float64
    tf_guess::Float64

    DescentProblem(;g=1.0,mdry=1.0,mwet=2.0,Tmin=2.0,Tmax=5.0,deltaMax=20.0,thetaMax=90.0,gammaGs=30.0,omMax=60.0,
                    jB=diagm([1e-2,1e-2,1e-2])/5, alpha=0.01,rTB=[-1e-2,0,0],rIi=[2.0,2.0,0.0],rIf=[0.0,0.0,0.0],
                    vIi=[-2,-2,0],vIf=[-0.1,0.0,0.0],qBIi=[1.0,0,0,0],qBIf=[1.0,0,0,0],wBi=[0.0,0.0,0.0],
                    wBf=[0.0,0,0],K=50,imax=15,wNu=1e5,wID=1e2, wDS=1e-3, nuTol=1e-10, delTol = 1e-3, tf_guess=5.0) =
        new(g,mdry,mwet,Tmin,Tmax,deltaMax,thetaMax,gammaGs,omMax,jB,alpha,rTB,rIi,rIf,vIi,vIf,
            qBIi,qBIf,wBi,wBf,K,imax,wNu,wID,wDS,nuTol,delTol,tf_guess)
end

struct ProbInfo
    a::Float64
    g0::Float64
    jB::SArray{Tuple{3,3},Float64,2,9}
    jBi::SArray{Tuple{3,3},Float64,2,9}
    rTB::SArray{Tuple{3}, Float64, 1, 3}
    ProbInfo(from::DescentProblem) = new(from.alpha, from.g, SMatrix{3,3}(from.jB), SMatrix{3,3}(inv(from.jB)), SVector{3}(from.rTB))
end

struct LinPoint
    state::MArray{Tuple{14}, Float64, 1, 14}
    control::MArray{Tuple{3}, Float64, 1, 3}
end

struct LinRes
    Ak::Array{Float64,2}
    Bk::Array{Float64,2}
    Ck::Array{Float64,2}
    Sigk::Array{Float64,1}
    Zk::Array{Float64,1}
    iter_end::Array{Float64, 1} #dunno
end

type ProblemIteration
    problem::DescentProblem
    sigma::Float64
    about::Array{LinPoint,1}
    dynam::Array{LinRes,  1}
end

struct IntegratorParameters
    dt :: Float64
    uk :: MArray{Tuple{3}, Float64, 1, 3}
    up :: MArray{Tuple{3}, Float64, 1, 3}
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

function create_initial(problem::DescentProblem)
    K = problem.K
    initial_points = Array{LinPoint,1}(K+1)
    for k=1:K+1
        mk = (K+1-k)/(K+1) * problem.mwet + (k/(K+1))*problem.mdry
        rIk = (K+1-k)/(K+1) * problem.rIi + (k/(K+1))*problem.rIf
        vIk = (K+1-k)/(K+1) * problem.vIi + (k/(K+1))*problem.vIf

        rot = rotation_between([1,0,0], -vIk)
        qBIk = [rot.w, rot.x, rot.y, rot.z]
        TBk = mk*problem.g*[1,0,0]
        state_init = vcat(mk,rIk,vIk,qBIk,[0.0,0,0])
        control_init = [mk*problem.g,0,0]
        initial_points[k] = LinPoint(state_init, control_init)
    end
    linpoints = linearize_dynamics(initial_points, problem.tf_guess, 1.0/(K+1), ProbInfo(problem))
    return ProblemIteration(problem, problem.tf_guess, initial_points, linpoints)
end

function solve_step(iteration::ProblemIteration)
    m = Model(solver = MosekSolver(MSK_IPAR_LOG=0))
    prob = iteration.problem
    K = iteration.problem.K
    iterDynam = iteration.dynam
    iterAbout = iteration.about

    tggs = tand(prob.gammaGs)
    sqcm = sqrt((1-cosd(prob.thetaMax))/2)
    delMax = cosd(prob.deltaMax)

    @variable(m, x[1:state_dim,1:K+1])
    @variable(m, u[1:control_dim,1:K+1])
    @variable(m, sigma)
    @variable(m, nu[1:state_dim, 1:K+1])
    @variable(m, optEta[1:K+1])
    @variable(m, optDelta[1:K+1])
    @variable(m, optDeltaS)
    @variable(m, optEta1[1:K+1])
    @variable(m, optEta2[1:K+1])
    @variable(m, optEta3)
    @variable(m, optVar1)
    @variable(m, optVar2)

    @objective(m, Min, -x[1,K+1] + prob.wNu*optVar2 + prob.wID*optVar1 + prob.wDS*sum(optDeltaS))
    @constraint(m, norm(optDelta) <= optVar1)
    @constraint(m, norm(optEta) <= optVar2)

    #initial straightforward constraints
    @constraint(m, x[mass_idx,1] == prob.mwet)
    @constraint(m, x[r_idx_it,1] .== prob.rIi)
    @constraint(m, x[v_idx_it,1] .== prob.vIi)
    @constraint(m, x[qbi_idx_it,1] .== prob.qBIi)
    @constraint(m, x[omb_idx_it,1] .== prob.wBi)
    #final straightforward constraints
    @constraint(m, x[r_idx_it,K+1] .== prob.rIf)
    @constraint(m, x[v_idx_it,K+1] .== prob.vIf)
    @constraint(m, x[qbi_idx_it,K+1] .== prob.qBIf)
    @constraint(m, x[omb_idx_it,K+1] .== prob.wBf)
    @constraint(m, u[2:3,K+1] .== [0,0])

    #dynamics
    for i=1:K
        dk = iterDynam[i]
        @constraint(m, x[:,i+1] .== dk.Ak * x[:,i] + dk.Bk * u[:,i] + dk.Ck * u[:,i+1] + dk.Sigk*sigma + dk.Zk + nu[:, i])
    end
    for i=1:K+1
        #state constraints
        @constraint(m, prob.mdry <= x[mass_idx,i])
        @constraint(m, tggs*norm(x[3:4,i]) <= x[2,i])
        @constraint(m, norm(x[10:11,i]) <= sqcm)
        @constraint(m, norm(x[12:14,i]) <= deg2rad(prob.omMax))

        #control constraints
        @constraint(m, delMax*norm(u[1:3,i]) <= u[1,i])
        @constraint(m, norm(u[1:3,i]) <= prob.Tmax)
        #linearized lower bound
        Blin = iterAbout[i].control[1:3]
        Bnorm = norm(Blin)
        @constraint(m, prob.Tmin <= dot(Blin, u[1:3,i])/Bnorm)

#=
        alphaBase = aoaFn(iterAbout[i].state)
        alphaDx = Array{Float64,1}(14)
        aoaJcb(0, iterAbout[i].state, alphaDx)
        @constraint(m, alphaBase + dot(alphaDx, x[:,i] - iterAbout[i].state) <= -0.7)
=#

        #trust region
        c1= sum(map(x->x^2,iterAbout[i].state)) - optEta1[i]
        @constraint(m, norm([(1-2*dot(iterAbout[i].state,x[:,i])+c1)/2,x[:,i]...]) <= (1+2*dot(iterAbout[i].state,x[:,i])-c1)/2)

        c2= sum(map(x->x^2,iterAbout[i].control)) - optEta2[i]
        @constraint(m, norm([(1-2*dot(iterAbout[i].control,u[:,i])+c2)/2,u[:,i]...]) <= (1+2*dot(iterAbout[i].control,u[:,i])-c2)/2)

        @constraint(m, norm(nu[:,i]) <= optEta[i])
        @constraint(m, optEta1[i] + optEta2[i] <= optDelta[i])
    end

    #time trust region
    b3 = 2*sigma*iteration.sigma
    c3 = iteration.sigma^2 - optDeltaS
    @constraint(m, norm([(1-b3+c3)/2, sigma]) <= (1+b3-c3)/2)

    #GO!
    JuMP.solve(m)

    #results
    xSol = getvalue(x)
    uSol = getvalue(u)
    sigmaSol = getvalue(sigma)

    #make linear points
    traj_points = Array{LinPoint,1}(K+1)
    for k=1:K+1
        traj_points[k] = LinPoint(xSol[:,k], uSol[:,k])
    end

    #linearize dynamics and return
    linpoints = linearize_dynamics(traj_points, sigmaSol, 1.0/(K+1), ProbInfo(prob))
    return ProblemIteration(prob, sigmaSol, traj_points, linpoints), norm(getvalue(nu)[:,:]), norm(getvalue(optDelta))
end

function solve_problem(iprob::DescentProblem)
    prob = create_initial(iprob)
    cnu = Inf
    cdel = Inf
    iter = 1
    while (iprob.nuTol < cnu || iprob.delTol < cdel) && iter < iprob.imax
        println(cnu, "|", cdel, "|", iprob.nuTol < cnu, "|", iprob.delTol < cdel)
        prob,cnu,cdel = solve_step(prob)
        iter = iter+1
    end
    return prob,cnu,cdel
end