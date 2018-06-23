module Rocketland
using JuMP
using Mosek
using MathOptInterface
using MathOptInterfaceMosek
using Rotations
using StaticArrays
using RocketlandDefns
import Dynamics
import FirstRound


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
    initial_points,linpoints = FirstRound.solve_initial(problem)
    model = build_model(problem, K, linpoints, initial_points, problem.tf_guess)
    return model #return ProblemIteration(problem, problem.tf_guess, initial_points, linpoints, model, 0, Inf)
end

const MOI=MathOptInterface
const SA=MOI.ScalarAffineTerm
const VA=MOI.VectorAffineTerm

function build_model(prob, K, iterDynam, iterAbout, sigHat)
	model = MosekOptimizer(MSK_IPAR_LOG=1,MSK_IPAR_INFEAS_REPORT_AUTO=1)
	dcs = MOI.ConstraintIndex[]
	state_nuc = MOI.ConstraintIndex[]

	tggs = tand(prob.gammaGs)
	sqcm = sqrt((1-cosd(prob.thetaMax))/2)
	delMax = cosd(prob.deltaMax)

	#main state variables
	xv = reshape(MOI.addvariables!(model, state_dim*(K+1)), state_dim, K+1)
	uv = reshape(MOI.addvariables!(model, control_dim*(K+1)), control_dim, K+1)
	dxv = reshape(MOI.addvariables!(model, state_dim*(K+1)), state_dim, K+1)
	duv = reshape(MOI.addvariables!(model, control_dim*(K+1)), control_dim, K+1)
	nuv = reshape(MOI.addvariables!(model, state_dim*(K+1)), state_dim, K+1)
	dsig = MOI.addvariable!(model)

	# state helpers
	xvh = reshape(MOI.addvariables!(model, 7*(K+1)), 7, K+1)
	uvh = reshape(MOI.addvariables!(model, 3*(K+1)), 3, K+1)

	#relaxations
	deltaI = MOI.addvariable!(model)
	deltaIs = MOI.addvariables!(model,K+1)
	deltasv = MOI.addvariable!(model)
	nuSum = MOI.addvariable!(model)
	tlb_viol = MOI.addvariables!(model, K+1)
	tlb_sviol = MOI.addvariable!(model)

	txv = MOI.addvariables!(model, K+1)
	sxv = MOI.addvariables!(model, K+1)
	tsv = MOI.addvariable!(model)
	ssv = MOI.addvariable!(model)
	print(deltaIs[K+1])

	#helpers
	gs_h = MOI.addvariables!(model, K+1)
	sqcm_h = MOI.addvariables!(model, K+1)
	ommax_h = MOI.addvariables!(model, K+1)
	uf_h = MOI.addvariables!(model, K+1)
	ctrl_h = MOI.addvariables!(model, K+1)
	rkv = MOI.addvariable!(model)
	duvh = reshape(MOI.addvariables!(model, control_dim*(K+1)), control_dim, K+1)

	#objective
	# x[1,K+1] + prob.wNu*optVar2 + prob.wID*optVar1 + prob.wDS*sum(optDeltaS)
	MOI.set!(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
				    MOI.ScalarAffineFunction([SA(-1.0, xv[1,K+1]), SA(prob.wNu, nuSum), SA(prob.wID, deltaI), SA(prob.wNu, tlb_sviol), SA(prob.wDS, deltasv)],0.0))
	MOI.set!(model, MOI.ObjectiveSense(), MOI.MinSense)

	MOI.addconstraint!(model, MOI.VectorOfVariables([nuSum; reshape(nuv, length(nuv))]), MOI.SecondOrderCone(1+length(nuv)))
	#=
	MOI.addconstraint!(model, MOI.VectorAffineFunction(
		vcat([VA(i, SA(1.0, deltaIs[i])) for i=1:K+1], 
			 [VA(i, SA(-1.0, txv[i])) for i=1:K+1]),fill(0.0, K+1)), MOI.Zeros(K+1))
			 =#
	MOI.addconstraint!(model, MOI.VectorOfVariables([deltaI; dxv[:,:]...; duv[:,:]...]), MOI.SecondOrderCone(1+(state_dim+control_dim)*(K+1)))
	MOI.addconstraint!(model, MOI.VectorOfVariables([tlb_sviol; tlb_viol]), MOI.SecondOrderCone(1+(K+1)))

	#initial state constraints
	eq_veccons(vect, val) = map((vr,vl)->MOI.addconstraint!(model, MOI.SingleVariable(vr), MOI.EqualTo(vl)), vect, val)

	MOI.addconstraint!(model, MOI.VectorAffineFunction(
		vcat([[VA(i+(j-1)*state_dim, SA(1.0, xv[i,j])) for i=1:state_dim, j=1:K+1]...], 
			 [[VA(i+(j-1)*state_dim, SA(-1.0, dxv[i,j])) for i=1:state_dim, j=1:K+1]...]), 
		reshape([-iterAbout[j].state[i] for i=1:state_dim, j=1:K+1],state_dim*(K+1))), MOI.Zeros(state_dim*(K+1)))

	MOI.addconstraint!(model, MOI.VectorAffineFunction(
		vcat([[VA(i+(j-1)*control_dim, SA(1.0, uv[i,j])) for i=1:control_dim, j=1:K+1]...], 
			 [[VA(i+(j-1)*control_dim, SA(-1.0, duv[i,j])) for i=1:control_dim, j=1:K+1]...]), 
		reshape([-iterAbout[j].control[i] for i=1:control_dim, j=1:K+1],control_dim*(K+1))), MOI.Zeros(control_dim*(K+1)))

	MOI.addconstraint!(model, MOI.SingleVariable(xv[mass_idx,1]), MOI.EqualTo(prob.mwet))
	eq_veccons(xv[r_idx_it,1], prob.rIi)
	eq_veccons(xv[v_idx_it,1], prob.vIi)
	#eq_veccons(xv[qbi_idx_it,1],prob.qBIi)
	eq_veccons(xv[omb_idx_it,1], prob.wBi)

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
		drv = dk.derivative
		ep = dk.endpoint - abn.state
		del = [dxv[:,i]; duv[:,i]; duv[:,i+1]; dsig]
		vars = vcat([[map((coeff, vari) -> VA(r, SA(coeff, vari)), drv[r,:], del); VA(r, SA(1.0,nuv[r,i])); VA(r, SA(-1.0,dxv[r,i+1]))] for r in 1:state_dim]...)
		push!(dcs, MOI.addconstraint!(model, MOI.VectorAffineFunction(vars, collect(ep)), MOI.Zeros(state_dim)))
	end

	MOI.addconstraint!(model, MOI.VectorAffineFunction(vcat([VA(t, SA(1/tggs, xv[mass_idx,t])) for t=1:K+1], [VA(t, SA(-1.0, gs_h[t])) for t=1:K+1]), zeros(K+1)), MOI.Zeros(K+1))
	MOI.addconstraint!(model, MOI.VectorAffineFunction([VA(t, SA(1.0, sqcm_h[t])) for t=1:K+1], fill(-sqcm, K+1)), MOI.Zeros(K+1))
	MOI.addconstraint!(model, MOI.VectorAffineFunction([VA(t, SA(1.0, ommax_h[t])) for t=1:K+1], fill(-deg2rad(prob.omMax), K+1)), MOI.Zeros(K+1))

	MOI.addconstraint!(model, MOI.VectorAffineFunction([[VA(t, SA(1/delMax,uv[1,t])) for t=1:K+1]; [VA(t, SA(-1.0,uf_h[t])) for t=1:K+1]], zeros(K+1)), MOI.Nonnegatives(K+1)) # u[1,i]/delMax-uf_h[i] >= 0
	MOI.addconstraint!(model, MOI.VectorAffineFunction([VA(t, SA(1.0, uf_h[t])) for t=1:K+1], fill(-prob.Tmax, K+1)), MOI.Nonpositives(K+1)) # uf_h[i] - prob.Tmax <= 0

	normed = map(lin->lin.control/norm(lin.control), iterAbout)
	consts = convert(Array{Float64,1},vcat(normed...))
	println(size(consts))
	tcs = MOI.addconstraint!(model, MOI.VectorAffineFunction([reshape([VA(t, SA(normed[t][i], uv[i,t])) for i=1:control_dim, t=1:K+1], control_dim*(K+1));
															  [VA(t, SA(1.0, tlb_viol[t])) for t=1:K+1]], fill(-prob.Tmin, K+1)), MOI.Nonnegatives(K+1)) # prob.Tmin <= dot(Blin/Bnorm, u[1:3,i])

	inds = [collect(3:4); collect(10:11); collect(12:14)]
	for i=1:K+1
		#state constraints
		if i > 1 # avoid bounding the initial state
			MOI.addconstraint!(model, MOI.SingleVariable(xv[mass_idx,i]), MOI.GreaterThan(prob.mdry))
		end
		MOI.addconstraint!(model, MOI.VectorAffineFunction([map((i,v) -> VA(i, SA(1.0,v)), 1:10, [xv[inds,i]; uv[:,i]]); map((i,v) -> VA(i, SA(-1.0,v)), 1:10, [xvh[:,i]; uvh[:,i]])], fill(0.0,10)), MOI.Zeros(10))
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
	#control trust region
	#=
	rkc = MOI.addconstraint!(model, MOI.SingleVariable(rkv), MOI.EqualTo(prob.rK))

	linv = reshape(hcat([iterAbout[i].control for i=1:K+1]...), control_dim*(K+1))
	rshp = reshape(uv, length(uv))
	dkc = MOI.addconstraint!(model, MOI.VectorAffineFunction([collect(1:control_dim*(K+1)); collect(1:control_dim*(K+1))], 
		[reshape(uv, length(uv)); uvh2], [fill(1, control_dim*(K+1)); fill(-1.0, control_dim*(K+1))], -linv), MOI.Zeros(control_dim*(K+1)))
	MOI.addconstraint!(model, MOI.VectorOfVariables([rkv; uvh2]), MOI.SecondOrderCone(1+length(uv)))
		=#
	#time trust region: same gubbins (QCP -> SOCP via a rotated SOC constraint) 
	# Want: (sigma - sigHat)^2 <= deltasv <=> sigma^2 -2 sigma sigmaHat + sigHat^2 - deltasv <= 0
	# Q = id, a = -2 sigHat, b=-deltasv + sigHat^2 (look ma no vectors) <=>
	# t + -2 sigHat sigma - deltasv = -b_c (3)
	# 2*s*t >= ||sigma||^2 where s = 1 (4)
	rkc = MOI.addconstraint!(model, MOI.SingleVariable(rkv), MOI.EqualTo(0.0))
	MOI.addconstraint!(model, MOI.VectorAffineFunction([
		[VA(i+(j-1)*control_dim, SA(1.0,duv[i,j])) for i=1:control_dim, j=1:K+1]...,
		[VA(i+(j-1)*control_dim, SA(-1.0, duvh[i,j])) for i=1:control_dim, j=1:K+1]...], fill(0.0,(K+1)*control_dim)), MOI.Zeros((K+1)*control_dim))
	MOI.addconstraint!(model, MOI.VectorOfVariables([rkv; duvh[:,:]...]), MOI.SecondOrderCone(1+length(uv)))
	MOI.addconstraint!(model, MOI.SingleVariable(ssv), MOI.EqualTo(1.0))
	MOI.addconstraint!(model, MOI.VectorOfVariables([ssv, deltasv, dsig]), MOI.RotatedSecondOrderCone(3))
	model,xv
	#return ProblemModel(m, xv, uv, dxv, duv, dsv, nuv, nuNorm, delta_xvs, delta_uvs, dynamic_constraints, pointing_constraints)
end

const MOI=MathOptInterface
function solve_step(iteration::ProblemIteration)
	prob = iteration.problem
	K = prob.K
	xv = iteration.model.xv
	uv = iteration.model.uv
	dxv = iteration.model.dxv
	duv = iteration.model.duv
	dsv = iteration.model.dsv
	state_base = iteration.model.state_base
	control_base = iteration.model.control_base
	dynamic_constraints = iteration.model.dynamic_constraints
	pointing_constraints = iteration.model.pointing_constraints

	iterDynam = iteration.dynam
	iterAbout = iteration.about
	sigHat = iteration.sigma

	moimod = iteration.model.socp_model.moibackend
	for i=1:K+1
		st = iterAbout[i].state
		ct = iterAbout[i].control
		for j=1:state_dim
			#MOI.modifyconstraint!(moimod, state_base[j,i].index, MOI.EqualTo(st[j]))
		end
		for j=1:control_dim
			#MOI.modifyconstraint!(moimod, control_base[j,i].index, MOI.EqualTo(ct[j]))
		end
	end
	for i=2:K
		#pointing 
		normed = iterAbout[i].control/norm(iterAbout[i].control)
		for j=1:control_dim
			#MOI.modifyconstraint!(moimod, pointing_constraints[i-1].index, MOI.ScalarCoefficientChange(uv[j,i].index, -normed[j]))
		end
	end

	for i=1:K
		dk = iterDynam[i]
		deriv = iterDynam[i].derivative
		endpoint = iterDynam[i].endpoint - iterAbout[i+1].state
		cstrs = dynamic_constraints[i]
		vrs = [dxv[:,i];duv[:,i];duv[:,i+1];dsv]
		#deriv * vrs [...] = -endpoint
		for j=1:state_dim # go row-wise through the matrix
			for vi = 1:length(vrs)
				#MOI.modifyconstraint!(moimod, cstrs[j].index, MOI.ScalarCoefficientChange(vrs[vi].index, deriv[j,vi]))
			end
			#MOI.modifyconstraint!(moimod, cstrs[j].index, MOI.EqualTo(-endpoint[j]))
		end
	end

	JuMP.optimize(iteration.model.socp_model)
	xs = JuMP.resultvalue.(xv)
	us = JuMP.resultvalue.(uv)
	traj_points = Array{LinPoint,1}(K+1)
	ds = JuMP.resultvalue.(dsv)
	for k=1:K+1
		traj_points[k] = LinPoint(xs[:,k], us[:,k])
	end
	println(ds)
	#linpoints = Dynamics.linearize_dynamics(traj_points, prob.tf_guess, 1.0/(K+1), ProbInfo(prob))

	return xs, us

    #=
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


	#make linear points
	traj_points = Array{LinPoint,1}(K+1)
	for k=1:K+1
		traj_points[k] = LinPoint(xsol1[:,k], usol1[:,k])
	end

	if iteration.rk == Inf
		next_rk = prob.ri
	else
		infos = ProbInfo(prob)
		jK = iterAbout[K+1].state[1] + prob.wNu*sum(norm(iterAbout[k+1].state - Dynamics.predict_state(iterAbout[k].state, iterAbout[k].control, iterAbout[k+1].control, sigHat, 1.0/(K+1), infos),1) for k=1:K)
		jKp = xsol1[1,K+1] + prob.wNu*sum(norm(xsol1[:,k+1] - Dynamics.predict_state(xsol1[:,k], usol1[:,k], usol1[:,k+1], sigmasol1, 1.0/(K+1), infos),1) for k=1:K)
		lk = xsol1[1,K+1] + prob.wNu*sum(norm(nusol1,1))

		djk = jK - jKp
		dlk = jK - lk
		rhk = djk/dlk
		if rhk < prob.rh0
			println("reject $rhk $(iteration.rk)")
			return ProblemIteration(prob, iteration.sigma, iteration.about, iteration.dynam, iteration.model, iteration.iter+1, iteration.rk/prob.alph), deltaIsol, deltasol1
		else
			if rhk < prob.rh1
				next_rk = iteration.rk/problem.alph
			elseif prob.rh1 <= rhk && rhk < prob.rh2
				next_rk = iteration.rk
			else 
				next_rk = problem.bet*iteration.rk
			end
		end
	end

	#linearize dynamics and return
	linpoints = Dynamics.linearize_dynamics(traj_points, sigmasol1, 1.0/(K+1), ProbInfo(prob))

	return ProblemIteration(prob, sigmasol1, traj_points, linpoints, iteration.model, iteration.iter+1, next_rk), deltaIsol, deltasol1 #, nusol1
	=#
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