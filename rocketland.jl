module Rocketland
using Mosek
using MathOptInterface
using MosekTools
using Rotations
using StaticArrays
using LinearAlgebra
using ForwardDiff
using ..RocketlandDefns
import ..Dynamics
import ..FirstRound


#state indexing
const state_dim = 14
const control_dim = 5
const r_idx_it = 2:4
const v_idx_it = 5:7
const qbi_idx_it = 8:11
const omb_idx_it = 12:14
const acc_width = state_dim+control_dim*2+3
const acc_height = state_dim
const mass_idx = 1

function create_initial(problem::DescentProblem, linear_cache::LinearCache)
    K = problem.K
    initial_points,linpoints = FirstRound.linear_initial(problem, linear_cache)
    model = build_model(problem, K, linpoints, initial_points, problem.tf_guess)
    return ProblemIteration(problem, linear_cache, problem.tf_guess, initial_points, linpoints, model, 0, Inf, Inf)
end

const MOI=MathOptInterface
const SAF=MOI.ScalarAffineFunction
const VAF=MOI.VectorAffineFunction
const SA=MOI.ScalarAffineTerm
const VA=MOI.VectorAffineTerm
const VoV=MOI.VectorOfVariables
const SOC=MOI.SecondOrderCone

# SCvx defns 
# Lk = -1 x[1, K+1] + 1e4 * norm(nuv)
# Jk = -1 x[1, K+1] + 1e4 * norm(x[:, i+1] - act[i+1])

function build_model(prob, K, iterDynam, iterAbout, sigHat)
	model = MosekOptimizer(#=MSK_IPAR_LOG=0,=#MSK_IPAR_INFEAS_REPORT_AUTO=1,MSK_IPAR_BI_IGNORE_MAX_ITER=1,
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
	dsig = MOI.add_variable(model)
	nuv = reshape(MOI.add_variables(model, state_dim*(K+1)), state_dim, K+1)

	# trust regions
	Jvnu = MOI.add_variable(model)
	Jtr = MOI.add_variable(model)
	Jsig = MOI.add_variable(model)

	#objective
	MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
		SAF([SA(-1.0, xv[1, K+1]), SA(prob.wNu, Jvnu), SA(0.5, Jtr), SA(1.0, Jsig)],0.0))
	MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

	# couple the xs and us to the dxs and the dus
	# about + dx = x; about + du = u; etc
    # vcat(iterAbout[:].state...) + reshape(xv, state_dim*(K+1)) - reshape(xv, state_dim*(K+1)) = 0
    # vcat(iterAbout[:].control...) + reshape(duv, control_dim*(K+1)) - reshape(uv, control_dim*(K+1)) = 0
    state_base = MOI.add_constraint(model, VAF(VA.([1:state_dim*(K+1); 1:state_dim*(K+1)], 
    	[SA.(1.0, reshape(dxv, state_dim*(K+1))); SA.(-1.0, reshape(xv, state_dim*(K+1)))]), 
    	Array(vcat(getfield.(iterAbout[:], :state)...))), MOI.Zeros(state_dim*(K+1)))
    control_base = MOI.add_constraint(model, VAF(VA.([1:control_dim*(K+1); 1:control_dim*(K+1)], 
    	[SA.(1.0, reshape(duv, control_dim*(K+1))); SA.(-1.0, reshape(uv, control_dim*(K+1)))]), 
    	Array(vcat(getfield.(iterAbout[:], :control)...))), MOI.Zeros(control_dim*(K+1)))

    # build the trust regions
    MOI.add_constraint(model, VoV([Jvnu; reshape(nuv, state_dim*(K+1))]), SOC(state_dim*(K+1) + 1))
    MOI.add_constraint(model, VoV([Jtr; reshape(dxv, state_dim*(K+1)); reshape(duv, control_dim*(K+1))]), SOC(1+state_dim*(K+1)+control_dim*(K+1)))
    MOI.add_constraint(model, VoV([Jsig; dsig]), SOC(2))

	#initial and final state constraints
	# r_1 = r_i, v_1 = v_i, omb_1 = omb_i; same for final
	# we assume that the base solution satisfies the required state constraints. Therefore, it is sufficient
	# to lock the differences to 0
	# done as one big vector equality constraint; 1 * vars - vals = 0 -> vars = vals
	vars = map(t -> VA(t...), enumerate(
		SA.(1.0, [xv[mass_idx, 1]; xv[r_idx_it,1]; xv[v_idx_it,1]; xv[omb_idx_it,1];
				  xv[r_idx_it,K+1]; xv[v_idx_it,K+1]; xv[qbi_idx_it,K+1]; xv[omb_idx_it,K+1];
				  uv[2:3, K+1]])))
	vals = -[prob.mwet; prob.rIi; prob.vIi; prob.wBi;
			            prob.rIf; prob.vIf; prob.qBIf; prob.wBf; 0.0; 0.0]
	MOI.add_constraint(model, VAF(vars, vals), MOI.Zeros(length(vars)))

	# linearized dynamics
	# where inp_n = [state_n; control_n; control_n+1; sigma]
	# at each step, ostate_{n+1} + dstate_{n+1} = linearized*(lininp_n + dinp_n) + nu
	# at each step, ostate_{n+1} + dstate_{n+1} = endpoint + linearized*dinp_n + nu
	# at each step, dstate_{n+1} = linearized*dinp_n + nu + (endpoint - ostate_{n+1})
	# so linearized*[dstate_n; dcontrol_n; dcontrol_n+1; dsigma] + nu + (endpoint - ostate_{n+1}) - dstate_{n+1} = 0
	dcstrs = MOI.ConstraintIndex[]
	for n=1:K
		matmul = vcat(map((col,var) -> vcat(map(x->VA(x...), enumerate(map(x -> SA(x,var), col)))...), 
					      eachcol(iterDynam[n].derivative), [dxv[:,n]; duv[:,n]; duv[:,n+1]; dsig])...)
		nu_eqn = map(invar -> VA(invar[1], SA(1.0, invar[2])), enumerate(nuv[:,n+1]))
		eq_eqn = map(invar -> VA(invar[1], SA(-1.0, invar[2])), enumerate(dxv[:,n+1]))
		lin_err = Array(iterDynam[n].endpoint - iterAbout[n+1].state)
		push!(dcstrs, MOI.add_constraint(model, VAF(vcat(matmul, nu_eqn, eq_eqn), lin_err), MOI.Zeros(state_dim)))
	end

	# state constraints wooo; equation nums from Szmuk and Acikmese 2018
	# eq. 6; mdry <= mk
	MOI.add_constraint(model, VAF(VA.(1:K, SA.(1.0, xv[1,2:K+1])), fill(-prob.mdry, K)), MOI.Nonnegatives(K))

	# eq. 7; tan (gamma_gs) ||rIk[2:3]||_2 <= rIk[1]
	# we model as ||rIk[2:3]||_2 <= (gshelp: rIk[1]/tan(gamma_gs))
	# initalize gshelp_k - 1/tan(gamma_ga) * rIk[1] = 0
	gshelp = MOI.add_variables(model, K)
	MOI.add_constraint(model, 
		VAF([VA.(1:K, SA.(1.0, gshelp)); VA.(1:K, SA.(-1.0/tggs, xv[r_idx_it[1], 1:K]))], zeros(K)), MOI.Zeros(K))
	# SECOND ORDER CONE WOOOO; ||rIk[2:3]||_2 <= gshelp_k
	for n=1:K
		MOI.add_constraint(model, VoV([gshelp[n]; xv[r_idx_it[2:3], n]]), SOC(3))
	end

	# eq. 28; cos thmax <= 1 - 2 ||qBIk[3:4]||^2_2
	# or ||qBIk[3:4]||^2_2  <= (1 - cos thmax)/2
	# or ||qBIk[3:4]||_2  <= (aoa_help: sqrt((1 - cos thmax)/2))
	# WE DON'T NEED NO STINKIN' ROTATED CONES
	# initialize aoa_help; need one for each cone :(
	aoa_help = MOI.add_variables(model, K)
	MOI.add_constraint(model, VAF(VA.(1:K, SA.(1.0, aoa_help)), fill(-sqcm, K)), MOI.Zeros(K))
	# ||qBIk[3:4]||_2 <= aoa_help
	for n=1:K
		MOI.add_constraint(model, VoV([aoa_help[n]; xv[qbi_idx_it[3:4], n]]), SOC(3))
	end

	# eq. 10; ||ombk||_2 <= (ang_sp_help: ommax)
	ang_sp_help = MOI.add_variables(model, K)
	MOI.add_constraint(model, VAF(VA.(1:K, SA.(1.0, ang_sp_help)), fill(-prob.omMax, K)), MOI.Zeros(K))
	for n=1:K
		MOI.add_constraint(model, VoV([ang_sp_help[n]; xv[omb_idx_it, n]]), SOC(4))
	end

	# constraints now from Szmuk, Reynolds, and Acikmese 2018 https://arxiv.org/pdf/1811.10803.pdf
	# max thrust constraint ||uk|| <= tMax (8)
	# max_thrust_vk = max_thrust
	#max_thrust = MOI.add_variables(model, 	K+1)
	#MOI.add_constraint(model, VAF(VA.(1:K+1, SA.(1.0, max_thrust)), fill(-prob.Tmax, K+1)), MOI.Zeros(K+1))
	# thrust angle constraint (9)
	# ||uk|| <= uk[3]/(cos delMax)
	# thrust_anglek - uk[3]/(cos delMax) = 0
	#thrust_angle = MOI.add_variables(model, K+1)
	#MOI.add_constraint(model, VAF([VA.(1:K+1, SA.(1.0, thrust_angle)); VA.(1:K+1, SA.(-1/delMax, uv[3,:]))], zeros(K+1)), MOI.Zeros(K+1))

	# combined thrust constraint 
	# ||uk|| <= min(max_thrust_vk, thrust_anglek)
	# mtk <= max_thrust_vk /\ mtk <= thrust_anglek
	# ||uk|| <= mtk
	mtk = MOI.add_variables(model, K+1)
	# mtk <= max_thrust_vk <=> mtk - max_thrust_vk <= 0 <=> mtk - prob.Tmax <= 0
	MOI.add_constraint(model, VAF(VA.(1:K+1, SA.(1.0,mtk)), fill(-prob.Tmax, K+1)), MOI.Nonpositives(K+1))
	# mtk <= thrust_anglek <=> mtk - thrust_anglek <= 0 <=> mtk - uk[3]/(cos delMax) <= 0
	MOI.add_constraint(model, VAF([VA.(1:K+1, SA.(1.0,mtk)); VA.(1:K+1, SA.(-1/delMax, uv[1,:]))], zeros(K+1)), MOI.Nonpositives(K+1))

	for n=1:K+1
		MOI.add_constraint(model, VoV([mtk[n]; uv[1:3, n]]), SOC(4))
	end

	#linearized thrust lower bound constraint (8 & 35)
	# Tmin - ||uk|| <= 0
	# h_tlb,k + H_tlb,k duk <= 0
	# h_tlb,k = Tmin - ||uk_lin||
	# H_tlb,k = d(Tmin-||uk_lin||)/duk_lin = -d(uk_lin)/duk_lin = -uk_lin.../norm(uk_lin)
	tlbvars = vcat((VA.(fill(n,3), SA.(-(iterAbout[n].control[1:3] ./ norm(iterAbout[n].control[1:3])), duv[1:3,n])) for n=1:K+1)...)
	tlbconsts = [(prob.Tmin - norm(iterAbout[n].control)) for n=1:K+1]
	thrust_lb_constraint = MOI.add_constraint(model, VAF(tlbvars, tlbconsts), MOI.Nonpositives(K+1))

	# simple fin constraints 
	finmxf = MOI.add_variables(model, K+1)
	#MOI.add_constraint(model, VAF(VA.(1:K+1, SA.(1.0, finmxf)), fill(-0.01,K+1)), MOI.Zeros(K+1))
	for n=1:K+1
		MOI.add_constraint(model, VAF(VA.(1:2, SA.(1.0, uv[4:5, n])), fill(-0.000001,2)), MOI.Nonpositives(2))
		#MOI.add_constraint(model, VoV([finmxf[n]; uv[4:5, n]]), SOC(3))
	end

	# state triggered constraints
	# todo 

	# SCvx trust region
	rK = MOI.add_variable(model)
	rKc = MOI.add_constraint(model, VAF([VA(1, SA(1.0, Jtr))], [-1.0]), MOI.Nonpositives(1))

	return ProblemModel(model, xv, uv, dxv, duv, dsig, nuv, rK, state_base, control_base, dcstrs, thrust_lb_constraint, rKc, (Jtr, ))
end

const MOI=MathOptInterface
function vsq_sub_dotsq(inp::Vector{T}) where T
	return dot(inp[1:3], Dynamics.DCM(inp[4:7])*[1.0,0,0])^2
end

function solve_step(iteration::ProblemIteration, linear_cache::LinearCache)
	prob = iteration.problem
	itermodel = iteration.model
	model = itermodel.socp_model
	K = prob.K
	xv = itermodel.xv
	uv = itermodel.uv
	dxv = itermodel.dxv
	duv = itermodel.duv
	dsig = itermodel.dsv
	nuv = itermodel.nuv
	rk = itermodel.rk
	dbg = itermodel.debug
	# fix the base constraints

	about = iteration.about
	dynam = iteration.dynam
	MOI.modify(model, itermodel.state_base, 
		MOI.VectorConstantChange(Array(vcat(getfield.(about[:], :state)...))))
	MOI.modify(model, itermodel.control_base, 
		MOI.VectorConstantChange(Array(vcat(getfield.(about[:], :control)...))))

	# fix the dynamic constraints
	for n=1:K
		vars = [dxv[:,n]; duv[:,n]; duv[:,n+1]; dsig]
		derivs = map(x -> collect(enumerate(x)), eachcol(dynam[n].derivative))
		changes = MOI.MultirowChange.(vars, derivs)
		map(x->MOI.modify(model, itermodel.dynamic_constraints[n], x), changes)
		lin_err = Array(dynam[n].endpoint - about[n+1].state)
		MOI.modify(model, itermodel.dynamic_constraints[n], MOI.VectorConstantChange(lin_err))
	end

	# fix the thrust lower bound
	mrcs = vcat((MOI.MultirowChange.(duv[1:3, n], 
				 map(x->[(n, x)], -about[n].control[1:3] ./ norm(about[n].control[1:3]))) for n=1:K+1)...)
	ncsts = [(prob.Tmin - norm(about[n].control[1:3])) for n=1:K+1]
	map(cstr -> MOI.modify(model, itermodel.thrust_lb_constraint, cstr), mrcs)
	MOI.modify(model, itermodel.thrust_lb_constraint, MOI.VectorConstantChange(ncsts))

	#update the trust region 
	println("rhk $(iteration.rk)")
	MOI.modify(model, itermodel.rkc, MOI.VectorConstantChange([-iteration.rk]))

	MOI.optimize!(model);

	status = MOI.get(model, MOI.TerminationStatus())
	if status != MOI.OPTIMAL
	    error("Non-optimal result $status exiting")
	end

	xr = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(xv, state_dim*(K+1))), state_dim, K+1)
	ur = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(uv, control_dim*(K+1))), control_dim, K+1)
	dsr = MOI.get(model, MOI.VariablePrimal(), dsig)
	nur = (reshape(MOI.get(model, MOI.VariablePrimal(), reshape(nuv, state_dim*(K+1))), state_dim, K+1))
	dxvr = (reshape(MOI.get(model, MOI.VariablePrimal(), reshape(dxv, state_dim*(K+1))), state_dim, K+1))
	Jtrr = MOI.get(model, MOI.VariablePrimal(), dbg[1])
	println("Jtr $Jtrr")

	info = ProbInfo(prob)
	# Lk = -1 x[1, K+1] + prob.wNu * norm(nuv)
	# Jk = -1 x[1, K+1] + prob.wNu * norm(x[:, i+1] - act[i+1])
	jK = -1 * xr[1, K+1] + prob.wNu * norm(xr[:,k+1] - Dynamics.predict_state(xr[:,k], ur[:,k], ur[:,k+1], iteration.sigma + dsr, 1.0/(K+1), info, linear_cache) for k=1:K)
	lK = -1 * xr[1, K+1] + prob.wNu * norm(nur)

	if iteration.rk == Inf
		next_rk = prob.ri
	else
		jKm = iteration.cost
		djk = jKm - jK
		dlk = jKm - lK
		rhk = djk/dlk
		if rhk < prob.rh0
			println("reject $rhk")
			return ProblemIteration(prob, iteration.cache, iteration.sigma, about, dynam, iteration.model, iteration.iter+1, iteration.rk/prob.alph, iteration.cost), dxvr
		elseif rhk < prob.rh1
			next_rk = iteration.rk/prob.alph
			case = 1
		elseif prob.rh1 <= rhk && rhk < prob.rh2
			next_rk = iteration.rk
			case = 2
		else 
			next_rk = prob.bet*iteration.rk
			case = 3
		end
		println("nrhk $rhk case $case")
	end

	traj_points = [LinPoint(xr[:,n], ur[:,n]) for n=1:K+1]
	#return traj_points
	nsig = iteration.sigma + dsr 
	linpoints = Dynamics.linearize_dynamics_symb(traj_points, iteration.sigma + dsr, linear_cache)
	return ProblemIteration(prob, iteration.cache, iteration.sigma + dsr, traj_points, linpoints, 
		iteration.model, iteration.iter+1, next_rk, jK), dxvr				

	#=
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
	xs = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(dxv, length(xv))), state_dim, K+1)
	us = reshape(MOI.get(model, MOI.VariablePrimal(), reshape(duv, length(uv))), control_dim, K+1)
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
	=#
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
		dp = dot(Dynamics.DCM(pt.state[qbi_idx_it])*[1.0,0,0], pt.state[v_idx_it])/norm(pt.state[v_idx_it])
		println(dp)
		dv = Dynamics.DCM(pt.state[qbi_idx_it])	*[1.0,0,0]
		xls = [pt.state[3] pt.state[4]; pt.state[3]+dv[2]/3 pt.state[4]+dv[3]/3]
		yls = [pt.state[2] pt.state[2]; pt.state[2]+dv[1]/3 pt.state[2]+dv[1]/3]
		plot!(p,xls,yls, layout=2)
	end
	display(p)
end
export solve_problem, DescentProblem
end