module FirstRound
using JuMP
using RocketlandDefns
using Mosek

function solve_initial(prob::DescentProblem)
    #computed constants
    kf = prob.K
    N = kf
    a = prob.alpha
    tfs = 4
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
    wmf = 1

    m = Model(solver=MosekSolver())

    @variable(m, u[1:3,1:N+1])
    @variable(m, s[1:N+1])
    @variable(m, r[1:3,1:N+1])
    @variable(m, rd[1:3,1:N+1])
    @variable(m, z[1:N+1])

    #initial pos
    @constraint(m, r[i=1:3,1] .== r0[i])
    @constraint(m, rd[i=1:3,1] .== v0[i])
    @constraint(m, r[1:3,N+1] .== 0)
    @constraint(m, rd[1:3,N+1] .== 0)
    @constraint(m, 0 == u[2,N+1])
    @constraint(m, 0 == u[3,N+1])
    @constraint(m, z[1] == log(m0))

    @objective(m, Min, -z[N+1])

    for i = 2:N+1
        for j = 1:3
            @constraint(m, r[j,i] == r[j,i-1] + dt/2 * (rd[j,i-1] + rd[j,i]) - dt^2/12 * (u[j,i] - u[j,i-1]))
        end
    end

    @constraint(m, rd[1,i=2:N+1] .== rd[1,i-1] + dt/2 * (u[1,i-1] + u[1,i]) - g0*dt)
    @constraint(m, rd[2,i=2:N+1] .== rd[2,i-1] + dt/2 * (u[2,i-1] + u[2,i]))
    @constraint(m, rd[3,i=2:N+1] .== rd[3,i-1] + dt/2 * (u[3,i-1] + u[3,i]))
    @constraint(m, z[i=2:N+1] .== z[i-1] - a*dt/2 * (s[i-1] + s[i]))
    for i = 1:N+1
        @constraint(m, norm(u[1:3,i]) <= s[i])
    end

    z0 = log(m0 - a * tmax * (Array(1:N+1)-1) * dt)
    mu1 = tmin * exp(-z0[1:N+1])
    mu2 = tmax * exp(-z0[1:N+1])
    tggs = tand(gammags)

    for i = 1:N+1
        Ap = mu1[i]/2
        A = sqrt(Ap)
        b = -mu1[i]*(1 + z0[i])
        c = (mu1[i]*z0[i]^2)/2 + mu1[i]*z0[i] + mu1[i]

        @constraint(m, norm([(1 + b*z[i] - s[i] + c)/2, A*z[i]]) <= (1-b*z[i] + s[i] - c)/2)

        @constraint(m, 0 <= s[i])
        @constraint(m, s[i] <= mu2[i] * (1-(z[i] - z0[i])))

        #glideslope
        @constraint(m, tggs*norm(r[[2,3],i]) <= r[1,i]) 
    end

    @constraint(m, z0[i=1:N+1] .<= z[i])
    @constraint(m, z[i=1:N+1] .<= log(m0 - a * tmin * (i - 1) * dt))

    #SC modifications
    solve(m)
    println(getvalue(z)[N+1])
    return getvalue(u)
end
end