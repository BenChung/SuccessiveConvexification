module Rocketland
using JuMP
using Mosek
using MathOptInterface
using MathOptInterfaceMosek
using Rotations
using StaticArrays
using RocketlandDefns
import Dynamics

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

function create_initial(problem::DescentProblem)
    K = problem.K
    initial_points = Array{LinPoint,1}(K+1)
    for k=0:K
        mk = (K-k)/(K) * problem.mwet + (k/(K))*problem.mdry
        rIk = (K-k)/(K) * problem.rIi + (k/(K))*problem.rIf
        vIk = (K-k)/(K) * problem.vIi + (k/(K))*problem.vIf

        rot = rotation_between([1,0,0], -vIk)
        qBIk = @SVector [rot.w, rot.x, rot.y, rot.z]
        TBk = @SVector [mk*problem.g,0,0]
        state_init = vcat(mk,rIk,vIk,qBIk,(@SVector [0.0,0,0]))
        control_init = @SVector [mk*problem.g,0,0]
        initial_points[k+1] = LinPoint(state_init, control_init)
    end
    linpoints = Dynamics.linearize_dynamics(initial_points, problem.tf_guess, 1.0/(K+1), ProbInfo(problem))
    model = build_model(problem, K, linpoints, initial_points, problem.tf_guess)
    return ProblemIteration(problem, problem.tf_guess, initial_points, linpoints, model)
end

const MOI=MathOptInterface

function build_model(prob, K, iterDynam, iterAbout, sigHat)
	model = MosekOptimizer(MSK_IPAR_LOG=1)
	dcs = MOI.ConstraintIndex[]
	state_nuc = MOI.ConstraintIndex[]

	tggs = tand(prob.gammaGs)
	sqcm = sqrt((1-cosd(prob.thetaMax))/2)
	delMax = cosd(prob.deltaMax)

	#main state variables
	xv = reshape(MOI.addvariables!(model, state_dim*(K+1)), state_dim, K+1)
	uv = reshape(MOI.addvariables!(model, control_dim*(K+1)), control_dim, K+1)
	nuv = reshape(MOI.addvariables!(model, state_dim*(K+1)), state_dim, K+1)
	sigmav = MOI.addvariable!(model)

	# state helpers
	xvh = reshape(MOI.addvariables!(model, 7*(K+1)), 7, K+1)
	uvh = reshape(MOI.addvariables!(model, 3*(K+1)), 3, K+1)

	#relaxations
	deltaI = MOI.addvariable!(model)
	deltaIs = MOI.addvariables!(model,K+1)
	deltasv = MOI.addvariable!(model)
	nuSum = MOI.addvariable!(model)

	txv = MOI.addvariables!(model, K+1)
	sxv = MOI.addvariables!(model, K+1)
	tsv = MOI.addvariable!(model)
	ssv = MOI.addvariable!(model)

	#helpers
	gs_h = MOI.addvariables!(model, K+1)
	sqcm_h = MOI.addvariables!(model, K+1)
	ommax_h = MOI.addvariables!(model, K+1)
	uf_h = MOI.addvariables!(model, K+1)
	ctrl_h = MOI.addvariables!(model, K+1)
	rkv = MOI.addvariable!(model)
	uvh2 = MOI.addvariables!(model, 3*(K+1))

	#objective
	# x[1,K+1] + prob.wNu*optVar2 + prob.wID*optVar1 + prob.wDS*sum(optDeltaS)
	MOI.set!(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
				    MOI.ScalarAffineFunction([xv[1,K+1],nuSum, deltaI, deltasv],[-1.0,prob.wNu, prob.wID, prob.wDS],0.0))
	MOI.set!(model, MOI.ObjectiveSense(), MOI.MinSense)

	MOI.addconstraint!(model, MOI.VectorOfVariables([nuSum; reshape(nuv, length(nuv))]), MOI.SecondOrderCone(1+length(nuv)))
	MOI.addconstraint!(model, MOI.VectorOfVariables([deltaI; deltaIs]), MOI.SecondOrderCone(1+length(deltaIs)))

	#initial state constraints
	eq_veccons(vect, val) = map((vr,vl)->MOI.addconstraint!(model, MOI.SingleVariable(vr), MOI.EqualTo(vl)), vect, val)

	MOI.addconstraint!(model, MOI.SingleVariable(xv[mass_idx,1]), MOI.EqualTo(prob.mwet))
	eq_veccons(xv[r_idx_it,1], prob.rIi)
	eq_veccons(xv[v_idx_it,1],prob.vIi)
	#eq_veccons(xv[qbi_idx_it,1],prob.qBIi)
	eq_veccons(xv[omb_idx_it,1],prob.wBi)

	#final state constraints
	eq_veccons(xv[r_idx_it,K+1],prob.rIf)
	eq_veccons(xv[v_idx_it,K+1],prob.vIf)
	eq_veccons(xv[qbi_idx_it,K+1],prob.qBIf)
	eq_veccons(xv[omb_idx_it,K+1],prob.wBf)
	eq_veccons(uv[2:3,K+1], [0.0,0.0])

	#dynamics
	for i=1:K
		dk = iterDynam[i]
		ab = iterAbout[i]
		abn = iterAbout[i+1]
		push!(dcs, Dynamics.next_step(model, xv[:,i+1], dk, iterAbout[i], iterAbout[i+1], 
				    				         xv[:,i], uv[:,i], uv[:,i+1], sigmav, sigHat, nuv[:,i]))
	end

	MOI.addconstraint!(model, MOI.VectorAffineFunction([collect(1:K+1);collect(1:K+1)], [xv[mass_idx,:]; gs_h], [fill(1/tggs, K+1); fill(-1.0, K+1)], zeros(K+1)), MOI.Zeros(K+1))
	MOI.addconstraint!(model, MOI.VectorAffineFunction(collect(1:K+1), sqcm_h, fill(1.0,K+1), fill(-sqcm, K+1)), MOI.Zeros(K+1))
	MOI.addconstraint!(model, MOI.VectorAffineFunction(collect(1:K+1), ommax_h, fill(1.0,K+1), fill(-deg2rad(prob.omMax), K+1)), MOI.Zeros(K+1))

	MOI.addconstraint!(model, MOI.VectorAffineFunction([collect(1:K+1);collect(1:K+1)], [uv[1,:]; uf_h], 
													   [fill(1/delMax, K+1); fill(-1.0, K+1)], zeros(K+1)), MOI.Nonnegatives(K+1)) # u[1,i]/delMax-uf_h[i] >= 0
	MOI.addconstraint!(model, MOI.VectorAffineFunction(collect(1:K+1), uf_h, fill(1.0, K+1), fill(-prob.Tmax, K+1)), MOI.Nonpositives(K+1)) # uf_h[i] - prob.Tmax <= 0

	normed = map(lin->lin.control/norm(lin.control), iterAbout)
	consts = convert(Array{Float64,1},vcat(normed...))
	println(size(consts))
	tcs = MOI.addconstraint!(model, MOI.VectorAffineFunction(repeat(1:K+1,inner=3), reshape(uv, length(uv)), consts, fill(-prob.Tmin, K+1)), MOI.Nonnegatives(K+1)) # prob.Tmin <= dot(Blin/Bnorm, u[1:3,i])

	inds = [collect(3:4); collect(10:11); collect(12:14)]
	for i=1:K+1
		#state constraints
		if i > 1 # avoid bounding the initial state
			MOI.addconstraint!(model, MOI.SingleVariable(xv[mass_idx,i]), MOI.GreaterThan(prob.mdry))
		end
		MOI.addconstraint!(model, MOI.VectorAffineFunction([collect(1:10);collect(1:10)], [xv[inds,i]; uv[:,i]; xvh[:,i]; uvh[:,i]], 
			[fill(1.0,10);fill(-1.0,10)], fill(0.0,10)), MOI.Zeros(10))
		MOI.addconstraint!(model, MOI.VectorOfVariables([gs_h[i]; xvh[1:2,i]]), MOI.SecondOrderCone(3)) # xv[2,i]/tggs >= norm(xv[3:4,i], gs_h[i]=xv[2,i]/tggs
		MOI.addconstraint!(model, MOI.VectorOfVariables([sqcm_h[i]; xvh[3:4,i]]), MOI.SecondOrderCone(3)) # sqcm >= norm(xv[10:11,i])
		MOI.addconstraint!(model, MOI.VectorOfVariables([ommax_h[i]; xvh[5:7,i]]), MOI.SecondOrderCone(4)) # deg2rad(prob.omMax) >= norm(xv[12:14,i])

		#control constraint
		MOI.addconstraint!(model, MOI.VectorOfVariables([uf_h[i]; uvh[1:3,i]]), MOI.SecondOrderCone(4)) # uf_h[i] >= norm(u[1:3,i])
	end	
	#trust region
	# sum_i (state_i - x_i)^2 = sum_i x_i^2 - sum_i 2 x_i state_i + sum_i state_i^2) <= optEta1
	# In standard form: x' Q x + a' x + b <= 0
	# Q = id, a = [-2 state_i for i], b= -optEta1 + sum_i state_i^2
	# we recast this as a rotated second order cone
	# t + a' x = -b where t is a new var (1)
	# x' id x <= 2*t <=> 
	# 2*s*t >= ||x||^2 where s is a variable constrained to 1 (2)
	for i=1:K+1
		b = (sum(iterAbout[i].state .^ 2) + sum(iterAbout[i].control .^ 2))/2
		cs = MOI.addconstraint!(model, MOI.ScalarAffineFunction(
			[xv[:,i]; txv[i]; deltaIs[i]], 
			[-iterAbout[i].state; 1.0; -1/2], 0.0), 
			MOI.EqualTo(-b)) # eqn (1)
		push!(state_nuc,cs)
		MOI.addconstraint!(model, MOI.SingleVariable(sxv[i]), MOI.EqualTo(1.0))
		MOI.addconstraint!(model, MOI.VectorOfVariables([sxv[i]; txv[i]; xv[:,i]]), MOI.RotatedSecondOrderCone(2+state_dim))
	end
	#control trust region
	rkc = MOI.addconstraint!(model, MOI.SingleVariable(rkv), MOI.EqualTo(prob.rK))

	linv = reshape(hcat([iterAbout[i].control for i=1:K+1]...), control_dim*(K+1))
	rshp = reshape(uv, length(uv))
	dkc = MOI.addconstraint!(model, MOI.VectorAffineFunction([collect(1:control_dim*(K+1)); collect(1:control_dim*(K+1))], 
		[reshape(uv, length(uv)); uvh2], [fill(1, control_dim*(K+1)); fill(-1.0, control_dim*(K+1))], -linv), MOI.Zeros(control_dim*(K+1)))
	MOI.addconstraint!(model, MOI.VectorOfVariables([rkv; uvh2]), MOI.SecondOrderCone(1+length(uv)))

	#time trust region: same gubbins (QCP -> SOCP via a rotated SOC constraint) 
	# Want: (sigma - sigHat)^2 <= deltasv <=> sigma^2 -2 sigma sigmaHat + sigHat^2 - deltasv <= 0
	# Q = id, a = -2 sigHat, b=-deltasv + sigHat^2 (look ma no vectors) <=>
	# t + -2 sigHat sigma - deltasv = -b_c (3)
	# 2*s*t >= ||sigma||^2 where s = 1 (4)
	sc = MOI.addconstraint!(model, MOI.ScalarAffineFunction(MOI.VariableIndex[sigmav, tsv, deltasv], [-sigHat, 1.0, -0.5], 0.0), MOI.EqualTo(-sigHat^2/2))
	MOI.addconstraint!(model, MOI.VectorOfVariables([ssv, tsv, sigmav]), MOI.RotatedSecondOrderCone(3))
	MOI.addconstraint!(model, MOI.SingleVariable(ssv), MOI.EqualTo(1.0))
	return ProblemModel(model, dcs, tcs, state_nuc, sc, rkc, dkc, xv, uv, sigmav, nuSum, nuv, deltaI, txv)
end


function solve_step(iteration::ProblemIteration)
	prob = iteration.problem
	K = prob.K
	model = iteration.model.socp_model
	xv = iteration.model.xv
	uv = iteration.model.uv
	sigmav = iteration.model.sigmav
	nuSum = iteration.model.nuSum
	deltaI = iteration.model.deltaI
	txv = iteration.model.txv
	nuv = iteration.model.nuv

	iterDynam = iteration.dynam
	iterAbout = iteration.about
	sigHat = iteration.sigma
	dcs = iteration.model.dynamic_constraints
	for i=1:K
		dk = iterDynam[i]
		ab = iterAbout[i]
		abn = iterAbout[i+1]
		Dynamics.update_next_step(model, dcs[i],
			dk, iterAbout[i], iterAbout[i+1], xv[:,i], uv[:,i], uv[:,i+1], sigmav, sigHat)
	end


	normed = map(lin->lin.control/norm(lin.control), iterAbout)
	consts = convert(Array{Float64,1},vcat(normed...))
	indexes = repeat(1:51,inner=3)
	vars = reshape(uv, length(uv))
	println(size(consts))
	for i=1:length(vars)
		MOI.modifyconstraint!(model, iteration.model.thrust_constraint, MOI.MultirowChange(vars[i], [indexes[i]], [consts[i]]))
	end

	#update trust region
	# sum_i (state_i - x_i)^2 = sum_i x_i^2 - sum_i 2 x_i state_i + sum_i state_i^2) <= optEta1
	# In standard form: x' Q x + a' x + b <= 0
	# Q = id, a = [-2 state_i for i], b= -optEta1 + sum_i state_i^2
	# we recast this as a rotated second order cone
	# t + a' x = -b where t is a new var (1)
	# x' id x <= 2*t <=> 
	# 2*s*t >= ||x||^2 where s is a variable constrained to 1 (2)
	trcs = iteration.model.trust_constraints
	for i=1:K+1
		b = (sum(iterAbout[i].state .^ 2) + sum(iterAbout[i].control .^ 2))/2
		ix = trcs[i]
		for j=1:state_dim
			MOI.modifyconstraint!(model, ix, MOI.ScalarCoefficientChange(xv[j,i], -iterAbout[i].state[j]))
		end
		MOI.modifyconstraint!(model, ix, MOI.EqualTo(-b))
	end

	rkc = iteration.model.rk_constraint
	uc = iteration.model.u_constraint
	MOI.modifyconstraint!(model, rkc, MOI.EqualTo(prob.rK))
	linv= reshape(hcat([iterAbout[i].control for i=1:K+1]...), control_dim*(K+1))
	MOI.modifyconstraint!(model, uc, MOI.VectorConstantChange(-linv))
	#update time trust region: same gubbins (QCP -> SOCP via a rotated SOC constraint) 
	# Want: (sigma - sigHat)^2 <= deltasv <=> sigma^2 -2 sigma sigmaHat + sigHat^2 - deltasv <= 0
	# Q = id, a = -2 sigHat, b=-deltasv + sigHat^2 (look ma no vectors) <=>
	# t + -2 sigHat sigma - deltasv = -b_c (3)
	# 2*s*t >= ||sigma||^2 where s = 1 (4)
	sc = iteration.model.sigtr_constraint
	MOI.modifyconstraint!(model, sc, MOI.ScalarCoefficientChange(sigmav, -sigHat))
	MOI.modifyconstraint!(model,sc, MOI.EqualTo(-sigHat^2/2))

	MOI.optimize!(model)
	status = MOI.get(model, MOI.TerminationStatus())
	if status != MOI.Success
		println(status)
		return
	end	
	result_status = MOI.get(model, MOI.PrimalStatus())
	if result_status != MOI.FeasiblePoint
	    error("Solver ran successfully did not return a feasible point. The problem may be infeasible.")
	end
	xsol1 = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(xv, length(xv))), state_dim, K+1)
	usol1 = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(uv, length(uv))), control_dim, K+1)
	sigmasol1 = MOI.get(model, MOI.VariablePrimal(), sigmav)
	deltasol1 = MOI.get(model, MOI.VariablePrimal(), nuSum)
	deltaIsol = MOI.get(model, MOI.VariablePrimal(), deltaI)
	txvsol1 = MOI.get(model, MOI.VariablePrimal(), txv)
	nusol1 = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(nuv, length(nuv))), state_dim, K+1)

	infos = ProbInfo(prob)
	jK = iterAbout[K+1].state[1] + prob.wNu*sum(norm(iterAbout[k+1].state - Dynamics.predict_state(iterAbout[k].state, iterAbout[k].control, iterAbout[k+1].control, sigHat, 1.0/(K+1), infos),1) for k=1:K)
	jKp = xsol1[1,K+1] + prob.wNu*sum(norm(xsol1[:,k+1] - Dynamics.predict_state(xsol1[:,k], usol1[:,k], usol1[:,k+1], sigmasol1, 1.0/(K+1), infos),1) for k=1:K)
	lk = xsol1[1,K+1] + prob.wNu*sum(norm(nusol1,1))

	djk = jK - jKp
	dlk = jK - lk

	println("$djk $dlk $(djk/dlk)")

	#make linear points
	traj_points = Array{LinPoint,1}(K+1)
	for k=1:K+1
		traj_points[k] = LinPoint(xsol1[:,k], usol1[:,k])
	end

	#linearize dynamics and return
	linpoints = Dynamics.linearize_dynamics(traj_points, sigmasol1, 1.0/(K+1), ProbInfo(prob))
	return ProblemIteration(prob, sigmasol1, traj_points, linpoints, iteration.model), deltaIsol, deltasol1 #, nusol1
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