module Rocketland
using JuMP
using Mosek
using Rotations
using StaticArrays
using RocketlandDefns

#state indexing
const state_dim = 14
const control_dim = 3
const r_idx_it = 2:4
const v_idx_it = 5:7
const qbi_idx_it = 8:11
const omb_idx_it = 12:14
const acc_width = state_dim+control_dim*2+3
const acc_height = state_dim
const mass_idx = 1

include("dynamics.jl")

type ProblemIteration
    problem::DescentProblem
    sigma::Float64
    about::Array{LinPoint,1}
    dynam::Array{Dynamics.LinRes,1}
end
	

function create_initial(problem::DescentProblem)
    K = problem.K
    initial_points = Array{LinPoint,1}(K+1)
    for k=1:K+1
        mk = (K+1-k)/(K+1) * problem.mwet + (k/(K+1))*problem.mdry
        rIk = (K+1-k)/(K+1) * problem.rIi + (k/(K+1))*problem.rIf
        vIk = (K+1-k)/(K+1) * problem.vIi + (k/(K+1))*problem.vIf

        rot = rotation_between([1,0,0], -vIk)
        qBIk = @SVector [rot.w, rot.x, rot.y, rot.z]
        TBk = @SVector [mk*problem.g,0,0]
        state_init = vcat(mk,rIk,vIk,qBIk,(@SVector [0.0,0,0]))
        control_init = @SVector [mk*problem.g,0,0]
        initial_points[k] = LinPoint(state_init, control_init)
    end
    linpoints = Dynamics.linearize_dynamics(initial_points, problem.tf_guess, 1.0/(K+1), ProbInfo(problem))
    return ProblemIteration(problem, problem.tf_guess, initial_points, linpoints)
end

function solve_step(iteration::ProblemIteration)
    m = Model(solver = MosekSolver(MSK_IPAR_LOG=0))
    prob = iteration.problem
    K = iteration.problem.K
    iterDynam = iteration.dynam
    iterAbout = iteration.about
    sigHat = iteration.sigma

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
        ab = iterAbout[i]
        abn = iterAbout[i+1]
        @constraint(m, x[:,i+1] .== Dynamics.next_step(dk, iterAbout[i], iterAbout[i+1], x[:,i], u[:,i], u[:,i+1], sigma, sigHat, nu[:,i]))
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
    linpoints = Dynamics.linearize_dynamics(traj_points, sigmaSol, 1.0/(K+1), ProbInfo(prob))
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
export solve_problem, DescentProblem
end