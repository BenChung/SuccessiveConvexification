module FirstRound
using JuMP
using ..RocketlandDefns
using Mosek
using MathOptInterface
using Mosek
using StaticArrays
using Rotations
import ..Dynamics
const MOI=MathOptInterface

#=
function solve_initial(prob::DescentProblem)
    #computed constants
    kf = prob.K
    N = kf
    a = prob.alpha
    tfs = prob.tf_guess
    dt = tfs/kf
    mu = [((kf - k)/kf) * prob.mwet + (k/kf)*prob.mdry for k=0:kf]
    gamm0 = prob.Tmin
    g0 = prob.g
    mdry = prob.mdry
    tmin = prob.Tmin
    tmax = prob.Tmax
    thetamax = prob.thetaMax
    gammags = prob.gammaGs
    m0 = prob.mwet
    r0 = prob.rIi
    rf = prob.rIf
    v0 = prob.vIi
    vf = prob.vIf
    n0 = [1 0 0]
    nf = [1 0 0]
    nsc = 20
    wkar = 100.0

    tggs = tand(prob.gammaGs)
    sqcm = (cosd(prob.thetaMax))
    delMax = cosd(prob.deltaMax)

    m = Model(with_optimizer(MosekOptimizer,MSK_IPAR_INFEAS_REPORT_AUTO=1))

    @variable(m, T[1:3,1:N+1])
    @variable(m, s[1:N+1])
    @variable(m, r[1:3,1:N+1])
    @variable(m, v[1:3,1:N+1])
    @variable(m, ma[1:N+1])
    @variable(m, ga[1:N+1])
    @variable(m, kaR[1:N+1])
    @variable(m, kaRr[1:N+1])
    @variable(m, ar[1:3,1:N+1])
    @variable(m, nkaR)

    #initial pos
    @constraint(m, r[i=1:3,1] .== r0[i])
    @constraint(m, v[i=1:3,1] .== v0[i])
    @constraint(m, ma[1] .== m0)
    @constraint(m, r[1:3,N+1] .== 0)
    @constraint(m, v[1:3,N+1] .== 0)
    @constraint(m, 0 == T[2,N+1])
    @constraint(m, 0 == T[3,N+1])

    @objective(m, Min, -ma[N+1] + wkar*nkaR)
    @constraint(m, [nkaR, kaRr[:]...] in MOI.SecondOrderCone(N+2))
    @constraint(m, kaRr[:] .== kaR[:])

    for i=1:N
        @constraint(m, ma[i+1] == ma[i]- a*(ga[i] + ga[i+1])*dt/2.0)
        ak = T[:,i]/mu[i] + ar[:,i] + [-g0,0,0]
        akp = T[:,i+1]/mu[i+1] + ar[:,i+1] + [-g0,0,0]
        @constraint(m, r[j=1:3,i+1] .== r[j,i] + v[j,i]*dt + 1/3 * (ak[j] + 1/2*akp[j])*dt^2)
        @constraint(m, v[j=1:3,i+1] .== v[j,i] + 1/2 * (ak[j] + akp[j])*dt)
    end

    for i=1:N+1
        @constraint(m, mdry <= ma[i])
        @constraint(m, [r[1,i]/tggs, r[2,i], r[3,i]] in MOI.SecondOrderCone(3))
        @constraint(m, [ga[i], T[:,i]...] in MOI.SecondOrderCone(4))
        @constraint(m, tmin <= ga[i])
        @constraint(m, ga[i] <= tmax)
        @constraint(m, ga[i]*sqcm <= T[1,i])
        @constraint(m, [kaR[i], ar[:,i]...] in MOI.SecondOrderCone(4))
    end
    JuMP.optimize(m)

    mass = JuMP.resultvalue.(ma)
    ri = JuMP.resultvalue.(r)
    vi = JuMP.resultvalue.(v)
    Ti = JuMP.resultvalue.(T)

    initial_points = Array{LinPoint,1}(N+1)
    for k=1:N+1
        mk = mass[k]
        rIk = ri[:,k]
        vIk = vi[:,k]

        rot = rotation_between([1,0,0], -Ti[:,k])
        qBIk = @SVector [rot.w, rot.x, rot.y, rot.z]
        state_init = vcat(mk,rIk,vIk,qBIk,(@SVector [0.0,0,0]))
        control_init = @SVector [norm(Ti[:,k]),0,0]
        initial_points[k] = LinPoint(state_init, control_init)
    end

    return initial_points,Dynamics.linearize_dynamics(initial_points, prob.tf_guess, 1.0/(N+1), ProbInfo(prob))
end
=#

function linear_points(problem::DescentProblem)
    K = problem.K
    initial_points = Array{LinPoint,1}(UndefInitializer(), K+1)
    for k=0:K
        mk = (K-k)/(K) * problem.mwet + (k/(K))*problem.mdry
        rIk = (K-k)/(K) * problem.rIi + (k/(K))*problem.rIf
        vIk = (K-k)/(K) * problem.vIi + (k/(K))*problem.vIf

        rot = rotation_between([1,0,0], -vIk)
        qBIk = [rot.w, rot.x, rot.y, rot.z]
        TBk = [mk*problem.g,0,0]
        state_init = vcat(mk,rIk,vIk,qBIk,([0.0,0,0]))
        control_init = [mk*problem.g,0,0,0,0]
        initial_points[k+1] = LinPoint(state_init, control_init)
    end
    return initial_points
end

function linear_initial(problem::DescentProblem, cache::Dynamics.LinearCache)
    initial_points = linear_points(problem)
    return initial_points, #Dynamics.linearize_dynamics(initial_points, problem.tf_guess, 1/(problem.K+1), ProbInfo(problem))
        Dynamics.linearize_dynamics_symb(initial_points, problem.tf_guess, cache)
end

end