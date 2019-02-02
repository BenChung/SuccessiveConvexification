module Rocketland
using Mosek
using MathOptInterface
using MathOptInterfaceMosek
using Rotations
using StaticArrays
using LinearAlgebra
using ForwardDiff
using ..RocketlandDefns
import ..Dynamics
import ..FirstRound


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
    linear_cache = Dynamics.initalize_linearizer(ProbInfo(problem), 1.0/(problem.K+1))
    initial_points,linpoints = FirstRound.linear_initial(problem, linear_cache)
    model = build_model(problem, K, linpoints, initial_points, problem.tf_guess)
    return ProblemIteration(problem, linear_cache, problem.tf_guess, initial_points, linpoints, model, 0, Inf, Inf)
end

const MOI=MathOptInterface
const SA=MOI.ScalarAffineTerm
const VA=MOI.VectorAffineTerm

function build_model(prob, K, iterDynam, iterAbout, sigHat)
	model = MosekOptimizer(MSK_IPAR_LOG=1,MSK_IPAR_INFEAS_REPORT_AUTO=1,MSK_IPAR_BI_IGNORE_MAX_ITER=1,
				MSK_IPAR_INTPNT_MAX_ITERATIONS=10000)
	dcs = MOI.ConstraintIndex[]
	state_nuc = MOI.ConstraintIndex[]

	tggs = tand(prob.gammaGs)
	sqcm = sqrt((1-cosd(prob.thetaMax))/2)
	delMax = cosd(prob.deltaMax)
	cosma = cosd(45.0)

	#main state variables
	xv = reshape(MOI.add_variables(model, state_dim*(K+1)), state_dim, K+1)
	uv = reshape(MOI.add_variables(model, control_dim*(K+1)), control_dim, K+1)
	dxv = reshape(MOI.add_variables(model, state_dim*(K+1)), state_dim, K+1)
	duv = reshape(MOI.add_variables(model, control_dim*(K+1)), control_dim, K+1)
	nuv = reshape(MOI.add_variables(model, state_dim*(K+1)), state_dim, K+1)
	dsig = MOI.add_variable(model)

	# state helpers
	xvh = reshape(MOI.add_variables(model, 7*(K+1)), 7, K+1)
	uvh = reshape(MOI.add_variables(model, 3*(K+1)), 3, K+1)

	#relaxations
	deltaI = MOI.add_variable(model)
	deltasv = MOI.add_variable(model)
	nuSum = MOI.add_variable(model)
	tlb_viol = MOI.add_variables(model, K+1)
	tlb_sviol = MOI.add_variable(model)
	aoa_viol = MOI.add_variables(model, K+1)
	aoa_sviol = MOI.add_variable(model)
	thr_tot = MOI.add_variable(model)

	txv = MOI.add_variables(model, K+1)
	sxv = MOI.add_variables(model, K+1)
	tsv = MOI.add_variable(model)
	ssv = MOI.add_variable(model)

	#helpers
	gs_h = MOI.add_variables(model, K+1)
	sqcm_h = MOI.add_variables(model, K+1)
	ommax_h = MOI.add_variables(model, K+1)
	uf_h = MOI.add_variables(model, K+1)
	ctrl_h = MOI.add_variables(model, K+1)
	rkv = MOI.add_variable(model)
	rkv2 = MOI.add_variable(model)
	duvh = reshape(MOI.add_variables(model, control_dim*(K+1)), control_dim, K+1)
	dxvh = reshape(MOI.add_variables(model, state_dim*(K+1)), state_dim, K+1)

	#objective
	# x[1,K+1] + prob.wNu*optVar2 + prob.wID*optVar1 + prob.wDS*sum(optDeltaS)
	MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
				    MOI.ScalarAffineFunction([SA(prob.wCst, dsig), SA(prob.wNu, nuSum), #= SA(prob.wID, deltaI), SA(prob.wDS, deltasv), =# SA(prob.wTviol, tlb_sviol), SA(prob.wTviol, aoa_sviol)],0.0))
	MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

	sumv = MOI.add_variable(model)
	MOI.add_constraint(model, MOI.SingleVariable(sumv), MOI.EqualTo(0.5))
	MOI.add_constraint(model, MOI.VectorOfVariables([nuSum; sumv; reshape(nuv, length(nuv))]), MOI.RotatedSecondOrderCone(2+length(nuv)))
	#MOI.add_constraint(model, MOI.VectorOfVariables([nuSum; reshape(nuv, length(nuv))]), MOI.SecondOrderCone(1+length(nuv)))
	#=
	MOI.add_constraint(model, MOI.VectorAffineFunction(
		vcat([VA(i, SA(1.0, deltaIs[i])) for i=1:K+1], 
			 [VA(i, SA(-1.0, txv[i])) for i=1:K+1]),fill(0.0, K+1)), MOI.Zeros(K+1))
			 =#
			 println("vvar1 : $(deltaI)")
	MOI.add_constraint(model, MOI.VectorOfVariables([deltaI; dxv[:,:]...; duv[:,:]...]), 
								MOI.SecondOrderCone(1+(state_dim + control_dim)*(K+1)))
	#MOI.add_constraint(model, MOI.VectorOfVariables([deltasv, dsig]), MOI.SecondOrderCone(2))
	#MOI.add_constraint(model, MOI.VectorOfVariables([thr_tot; uv[:,:]...]), MOI.SecondOrderCone(1+(control_dim)*(K+1)))
	sumv3 = MOI.add_variable(model)
	MOI.add_constraint(model, MOI.SingleVariable(sumv3), MOI.EqualTo(0.5))

	MOI.add_constraint(model, MOI.VectorOfVariables([tlb_sviol; sumv3; tlb_viol]), MOI.RotatedSecondOrderCone(2+(K+1)))
	sumv4 = MOI.add_variable(model)
	MOI.add_constraint(model, MOI.SingleVariable(sumv4), MOI.EqualTo(0.5))
	MOI.add_constraint(model, MOI.VectorOfVariables([aoa_sviol; sumv4; aoa_viol]), MOI.RotatedSecondOrderCone(2+(K+1)))

	#initial state constraints
	eq_veccons(vect, val) = map((vr,vl)->MOI.add_constraint(model, MOI.SingleVariable(vr), MOI.EqualTo(vl)), vect, val)

	scr = MOI.add_constraint(model, MOI.VectorAffineFunction(
		vcat([[VA(i+(j-1)*state_dim, SA(1.0, xv[i,j])) for i=1:state_dim, j=1:K+1]...], 
			 [[VA(i+(j-1)*state_dim, SA(-1.0, dxv[i,j])) for i=1:state_dim, j=1:K+1]...]), 
		reshape([-iterAbout[j].state[i] for j=1:K+1, i=1:state_dim],state_dim*(K+1))), MOI.Zeros(state_dim*(K+1)))

	ccr = MOI.add_constraint(model, MOI.VectorAffineFunction(
		vcat([[VA(i+(j-1)*control_dim, SA(1.0, uv[i,j])) for i=1:control_dim, j=1:K+1]...], 
			 [[VA(i+(j-1)*control_dim, SA(-1.0, duv[i,j])) for i=1:control_dim, j=1:K+1]...]), 
		reshape([-iterAbout[j].control[i] for i=1:control_dim, j=1:K+1],control_dim*(K+1))), MOI.Zeros(control_dim*(K+1)))

	MOI.add_constraint(model, MOI.SingleVariable(xv[mass_idx,1]), MOI.EqualTo(prob.mwet))
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
		push!(dcs, MOI.add_constraint(model, MOI.VectorAffineFunction(vars, collect(ep)), MOI.Zeros(state_dim)))
	end

	MOI.add_constraint(model, MOI.VectorAffineFunction(vcat([VA(t, SA(1/tggs, xv[2,t])) for t=1:K+1], [VA(t, SA(-1.0, gs_h[t])) for t=1:K+1]), zeros(K+1)), MOI.Zeros(K+1))
	MOI.add_constraint(model, MOI.VectorAffineFunction([VA(t, SA(1.0, sqcm_h[t])) for t=1:K+1], fill(-sqcm, K+1)), MOI.Zeros(K+1))
	MOI.add_constraint(model, MOI.VectorAffineFunction([VA(t, SA(1.0, ommax_h[t])) for t=1:K+1], fill(-deg2rad(prob.omMax), K+1)), MOI.Zeros(K+1))

	MOI.add_constraint(model, MOI.VectorAffineFunction([[VA(t, SA(1/delMax,uv[1,t])) for t=1:K+1]; [VA(t, SA(-1.0,uf_h[t])) for t=1:K+1]], zeros(K+1)), MOI.Nonnegatives(K+1)) # u[1,i]/delMax >= uf_h[i]
	MOI.add_constraint(model, MOI.VectorAffineFunction([VA(t, SA(1.0, uf_h[t])) for t=1:K+1], fill(-prob.Tmax, K+1)), MOI.Nonpositives(K+1)) # uf_h[i] <= prob.Tmax

	normed = map(lin->lin.control/norm(lin.control), iterAbout)
	consts = convert(Array{Float64,1},vcat(normed...))
	
	tcs = MOI.add_constraint(model, MOI.VectorAffineFunction([reshape([VA(t, SA(0.0, uv[i,t])) for i=1:control_dim, t=1:K+1], control_dim*(K+1));
															  [VA(t, SA(1.0, tlb_viol[t])) for t=1:K+1]], fill(-prob.Tmin, K+1)), MOI.Nonnegatives(K+1)) # prob.Tmin <= dot(Blin/Bnorm, u[1:3,i]) + viol

	#non-convex
	# ||v||^2_2 - dot(Dynamics.DCM(xv[qbi_idx_it,i]) * [-1,0,0], xv[v_idx_it,i])^2 <= 2*maxDp/rho
	# q0 = quat[1]
	# q1 = quat[2]
	# q2 = quat[3]
	# q3 = quat[4]
	# qv1 = -(1-2*(q2^2 + q3^2))
	# qv2 = -(2*(q1 * q2 + q0 * q3))
	# qv3 = -(2*(q1 * q3 - q0 * q2))
	# ||xv[v_idx_it,i]||^2_2 <= sqV
	# sqV - [dot(q,v)^2|_q <= (2*maxDp/rho)^2
	# sqV - [dot(q,v)^2|_q <= (2*maxDp/rho)^2
	# sqV - [dot(q,v)^2|_q - (2*maxDp/rho)^2 <= 0
	# sqV - (base_{v,q} + grad*{dv,dq}) - (2*maxDp/rho)^2 <= 0
	# sqV - grad*{dv,dq} - (2*maxDp/rho)^2 - base_{v,q} <= 0
	sqVs = MOI.add_variables(model, K+1)
	oneHalves = MOI.add_variables(model, K+1)
	for i=1:K+1
		MOI.add_constraint(model, MOI.SingleVariable(oneHalves[i]), MOI.EqualTo(0.5))
		MOI.add_constraint(model, MOI.VectorOfVariables([sqVs[i]; oneHalves[i]; xv[v_idx_it,i]]), MOI.RotatedSecondOrderCone(5))
	end
	pcs = MOI.add_constraint(model, MOI.VectorAffineFunction([
		vcat(([VA(t, SA(0.0, v)) for v in xv[v_idx_it,t]] for t=1:K+1)...,
			 ([VA(t, SA(0.0, v)) for v in xv[qbi_idx_it,t]] for t=1:K+1)...); [VA(t, SA(1.0, sqVs[t])) for t=1:K+1]; [VA(t, SA(1.0, aoa_viol[t])) for t=1:K+1]], 
		fill(-(2*prob.dpMax/prob.rho)^2, K+1)), MOI.Nonpositives(K+1))

	for i=1:K+1
		MOI.add_constraint(model, MOI.SingleVariable(tlb_viol[i]), MOI.GreaterThan(0.0))
	end
	inds = [collect(3:4); collect(10:11); collect(12:14)]
	for i=1:K+1
		#state constraints
		if i > 1 # avoid bounding the initial state
			MOI.add_constraint(model, MOI.SingleVariable(xv[mass_idx,i]), MOI.GreaterThan(prob.mdry))
		end
		MOI.add_constraint(model, MOI.VectorAffineFunction([map((i,v) -> VA(i, SA(1.0,v)), 1:10, [xv[inds,i]; uv[:,i]]); map((i,v) -> VA(i, SA(-1.0,v)), 1:10, [xvh[:,i]; uvh[:,i]])], fill(0.0,10)), MOI.Zeros(10))
		MOI.add_constraint(model, MOI.VectorOfVariables([gs_h[i]; xvh[1:2,i]]), MOI.SecondOrderCone(3)) # xv[2,i]/tggs >= norm(xv[3:4,i], gs_h[i]=xv[2,i]/tggs
		MOI.add_constraint(model, MOI.VectorOfVariables([sqcm_h[i]; xvh[3:4,i]]), MOI.SecondOrderCone(3)) # sqcm >= norm(xv[10:11,i])
		MOI.add_constraint(model, MOI.VectorOfVariables([ommax_h[i]; xvh[5:7,i]]), MOI.SecondOrderCone(4)) # deg2rad(prob.omMax) >= norm(xv[12:14,i])

		#control constraint
		MOI.add_constraint(model, MOI.VectorOfVariables([uf_h[i]; uvh[1:3,i]]), MOI.SecondOrderCone(4)) # uf_h[i] >= norm(u[1:3,i])
	end	
	rkc = MOI.add_constraint(model, MOI.SingleVariable(rkv), MOI.LessThan(10.0))
	MOI.add_constraint(model, MOI.VectorAffineFunction([
		[VA(i+(j-1)*control_dim, SA(1.0,duv[i,j])) for i=1:control_dim, j=1:K+1]...;
		[VA(i+(j-1)*control_dim, SA(-1.0, duvh[i,j])) for i=1:control_dim, j=1:K+1]...], fill(0.0,(K+1)*control_dim)), MOI.Zeros((K+1)*control_dim))
	MOI.add_constraint(model, MOI.VectorAffineFunction([
		[VA(i+(j-1)*state_dim, SA(1.0,dxv[i,j])) for i=1:state_dim, j=1:K+1]...;
		[VA(i+(j-1)*state_dim, SA(-1.0, dxvh[i,j])) for i=1:state_dim, j=1:K+1]...], fill(0.0,(K+1)*state_dim)), MOI.Zeros((K+1)*state_dim))
	sumv2 = MOI.add_variable(model)
	MOI.add_constraint(model, MOI.SingleVariable(sumv2), MOI.EqualTo(0.5))
	MOI.add_constraint(model, MOI.VectorOfVariables([rkv; sumv2; duvh[:,:]...; dxvh[:,:]...; dsig]), MOI.RotatedSecondOrderCone(3+length(uv)+length(xv)))

	return ProblemModel(model, xv, uv, dxv, duv, dsig, nuv, tlb_viol, aoa_viol, rkv, scr, ccr, dcs, tcs, pcs, rkc)
end

const MOI=MathOptInterface
function vsq_sub_dotsq(inp::Vector{T}) where T
	return dot(inp[1:3], Dynamics.DCM(inp[4:7])*[1.0,0,0])^2
end

function solve_step(iteration::ProblemIteration)
	prob = iteration.problem
	K = prob.K
	xv = iteration.model.xv
	uv = iteration.model.uv
	dxv = iteration.model.dxv
	duv = iteration.model.duv
	dsig = iteration.model.dsv
	nuv = iteration.model.nuv
	rkv = iteration.model.rkv
	ctrl_viol = iteration.model.tnv
	aoa_viol = iteration.model.anv
	state_base = iteration.model.state_base
	control_base = iteration.model.control_base
	dynamic_constraints = iteration.model.dynamic_constraints
	pointing_constraint = iteration.model.pointing_constraints
	aoa_constraint = iteration.model.aoa_constraint
	trust_region = iteration.model.trust_region

	iterDynam = iteration.dynam
	iterAbout = iteration.about
	sigHat = iteration.sigma
	model = iteration.model.socp_model
	MOI.modify(model, state_base, MOI.VectorConstantChange(reshape([-iterAbout[j].state[i] for i=1:state_dim, j=1:K+1],state_dim*(K+1))))
	MOI.modify(model, control_base, MOI.VectorConstantChange(reshape([-iterAbout[j].control[i] for i=1:control_dim, j=1:K+1],control_dim*(K+1))))
	for i=1:K
		cstr = dynamic_constraints[i]
		derv = iterDynam[i].derivative
		ep = iterDynam[i].endpoint - iterAbout[i+1].state
		del = [dxv[:,i]; duv[:,i]; duv[:,i+1]; dsig]
		for j=1:length(del)
			vari = del[j]
			MOI.modify(model, cstr, MOI.MultirowChange(vari, [(r,derv[r,j]) for r = 1:state_dim]))
		end
		MOI.modify(model, cstr, MOI.VectorConstantChange(collect(ep)))
	end
	for i=1:K+1
		#pointing 
		normed = iterAbout[i].control/norm(iterAbout[i].control)
		for j=1:control_dim
			MOI.modify(model, pointing_constraint, MOI.MultirowChange(uv[j,i], [(i, normed[j])]))
		end
	end
	vidxes = vcat(v_idx_it,qbi_idx_it)
	ncons = Array{Float64,1}(undef, K+1)
	for i=1:K+1
		#aoa
		state = iterAbout[i].state[vidxes]
		vars = xv[vidxes,i]
		base = vsq_sub_dotsq(state)
		grad = -ForwardDiff.gradient(vsq_sub_dotsq, state)
		ncons[i] =  -(2*prob.dpMax/prob.rho)^2 - base
		for p in zip(vars,grad)
			MOI.modify(model, aoa_constraint, MOI.MultirowChange(p[1],[(i, p[2])]))
		end
	end
	MOI.modify(model, aoa_constraint, MOI.VectorConstantChange(ncons))

	println(iteration.rk)
	if iteration.rk != Inf
		MOI.set(model, MOI.ConstraintSet(), trust_region, MOI.LessThan(iteration.rk))
	end
	MOI.optimize!(model)
	status = MOI.get(model, MOI.TerminationStatus())
	if status != MOI.OPTIMAL
		println(status)
		if status != MOI.ITERATION_LIMIT
			return
		end
	else
		println("optimal found")
	end	
	result_status = MOI.get(model, MOI.PrimalStatus())
	if result_status != MOI.FEASIBLE_POINT
		println(result_status)
	    println("Solver ran successfully did not return a feasible point. The problem may be infeasible. Trying to continue.")
	end

	println(MOI.get(model, MOI.VariablePrimal(), rkv))
	xs = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(xv, length(xv))), state_dim, K+1)
	us = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(uv, length(uv))), control_dim, K+1)
	dxs = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(dxv, length(xv))), state_dim, K+1)
	dus = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(duv, length(uv))), control_dim, K+1)
	nus = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(nuv, length(nuv))), state_dim, K+1)
	dus = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(duv, length(duv))), control_dim, K+1)
	ds = MOI.get(model, MOI.VariablePrimal(), dsig)
	viol = MOI.get(model, MOI.VariablePrimal(), ctrl_viol)
	aviol = MOI.get(model, MOI.VariablePrimal(), aoa_viol)
	traj_points = Array{LinPoint,1}(undef, K+1)
	for k=1:K+1
		traj_points[k] = LinPoint(xs[:,k], us[:,k])
	end
	infos = ProbInfo(prob)
	pv = 0.0
	for k=1:K
		ps = Dynamics.predict_state(xs[:,k], us[:,k], us[:,k+1], sigHat + ds, 1.0/(K+1), infos)
		pv += norm(xs[:,k+1] - ps,1)
	end
	ic = norm(collect(min(norm(us[:,k])-prob.Tmin,0.0) for k=1:K+1),1)
	daoa = 0.0
	aoas = []
	for k=1:K+1
		quat = xs[qbi_idx_it,k]
		q0 = quat[1]
		q1 = quat[2]
		q2 = quat[3]
		q3 = quat[4]
		qv1 = -(1-2*(q2^2 + q3^2)) 
		qv2 = -(2*(q1 * q2 + q0 * q3))
		qv3 = -(2*(q1 * q3 - q0 * q2))
		cosaoa = min(dot([qv1,qv2,qv3], xs[v_idx_it,k])/(norm(quat)*norm(xs[v_idx_it,k])),1.0)
		push!(aoas, acosd(cosaoa))
		daoa += max(cosd(45.0) - cosaoa, 0)
	end
	println("costs ic:$(norm(viol,1)) daoa:$daoa rp:$(norm(aviol,1)) $(maximum(aoas)) $(iteration.rk)")
	cost = prob.wNu*pv + prob.wTviol*ic + prob.wTviol*daoa

	#linpoints = Dynamics.linearize_dynamics_rk4(traj_points, sigHat + ds, 1.0/(K+1), ProbInfo(prob))
	#linpoints = Dynamics.linearize_dynamics(traj_points, sigHat + ds, 1.0/(K+1), ProbInfo(prob))

	#=
	dstate = sum(norm(linpoints[i].endpoint - linpoints_rk4[i].endpoint) for i=1:K-1)
	djacob = sum(norm(linpoints[i].derivative .- linpoints_rk4[i].derivative) for i=1:K-1)
	println((linpoints[1].derivative .- linpoints_rk4[1].derivative) ./ (linpoints[1].derivative))
	println("dstate:$dstate djacob:$djacob")
	=#

	if iteration.rk == Inf
		next_rk = prob.ri
	else
		jK = iteration.cost
		lk = prob.wNu*sum(norm(nus[:,k],1) for k=1:K+1) + prob.wTviol*norm(viol,1) + prob.wTviol*norm(aviol,1)
		djk = jK - cost
		dlk = jK - lk
		rhk = djk/dlk
		if rhk < prob.rh0
			println("reject $rhk $djk $jK $(sum(norm(dus))) $(iteration.rk)")
			next_rk = 1.0
			return ProblemIteration(prob,  iteration.cache, iteration.sigma, iteration.about, iteration.dynam, iteration.model, iteration.iter+1, iteration.rk/prob.alph, iteration.cost)
		else
			case = 0
			if rhk < prob.rh1
				next_rk = iteration.rk/prob.alph
				case = 1
			elseif prob.rh1 <= rhk && rhk < prob.rh2
				next_rk = iteration.rk
				case = 2
			else 
				next_rk = prob.bet*iteration.rk
				case = 3
			end
			println("accept $rhk $cost pc:$dlk $rhk $case")
		end
	end
	linpoints = Dynamics.linearize_dynamics_symb(traj_points, sigHat + ds, iteration.cache)
	return ProblemIteration(prob, iteration.cache, sigHat + ds, traj_points, linpoints, iteration.model, iteration.iter+1, next_rk, cost)
end

function run_iters(iprob::DescentProblem, niters::Int)
	ip = Rocketland.create_initial(iprob)
	trjs = []
	tfs = []
	for i = 1:niters
		ip = solve_step(ip)
		push!(tfs, ip.sigma)
		push!(trjs, hcat([pt.state[2:4] for pt in ip.about]...))
	end
	return trjs, tfs
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

using Plots
using StaticArrays
function plot_solution(ip::ProblemIteration)
	ys = hcat(([pt.state[2],pt.state[2]] for pt in ip.about)...)'
	xs = hcat(([pt.state[3],pt.state[4]] for pt in ip.about)...)'
	tmax = max(maximum(pt.state[2] for pt in ip.about),
			maximum(pt.state[3] for pt in ip.about))
	pmax = max(maximum(pt.state[2] for pt in ip.about),
			maximum(pt.state[4] for pt in ip.about))
	tmin = min(minimum(pt.state[2] for pt in ip.about),
			minimum(pt.state[3] for pt in ip.about))
	pmin = min(minimum(pt.state[2] for pt in ip.about),
			minimum(pt.state[4] for pt in ip.about))

	thr = [norm(pt.control)/ip.problem.Tmax for pt in ip.about]
	println(thr)
	p=plot(xs,ys, xlims = (tmin,tmax), layout=2, legend=false)
	for pt in ip.about
		dp = dot(Dynamics.DCM(SVector{4}(pt.state[qbi_idx_it]))*[1.0,0,0], pt.state[v_idx_it])/norm(pt.state[v_idx_it])
		println(dp)
		dv = Dynamics.DCM(SVector{4}(pt.state[qbi_idx_it]))*[1.0,0,0]
		xls = [pt.state[3] pt.state[4]; pt.state[3]+dv[2]/3 pt.state[4]+dv[3]/3]
		yls = [pt.state[2] pt.state[2]; pt.state[2]+dv[1]/3 pt.state[2]+dv[1]/3]
		plot!(p,xls,yls, layout=2)
	end
	display(p)
end
export solve_problem, DescentProblem
end