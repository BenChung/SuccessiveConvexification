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
    return model
    return ProblemIteration(problem, problem.tf_guess, initial_points, linpoints, model, 0, Inf)
end

const MOI=MathOptInterface
const SAT=MOI.ScalarAffineTerm

function build_model(prob, K, iterDynam, iterAbout, sigHat)
	m = JuMP.Model(optimizer=MosekOptimizer(MSK_IPAR_INFEAS_REPORT_AUTO=1))
	tggs = tand(prob.gammaGs)
	sqcm = (1-cosd(prob.thetaMax))/2
	delMax = cosd(prob.deltaMax)

	#@variable(m, dxv[1:state_dim, 1:K+1])
	#@variable(m, duv[1:control_dim, 1:K+1])
	@variable(m, xv[1:state_dim, 1:K+1])
	@variable(m, uv[1:control_dim, 1:K+1])
	@variable(m, sv)

	#@variable(m, nuv[1:state_dim, 1:K])
	@variable(m, thr[1:K+1])
	@variable(m, tot_thr)

	@variable(m, delk)
	@variable(m, dels)
	#@variable(m, nuNorm)

	# set up the delta and solution variables
	state_lin = hcat(((iterAbout[i].state) for i=1:K+1)...)
	control_lin = hcat(((iterAbout[i].control) for i=1:K+1)...)
	#delta_xvs = @constraint(m, xv[j=1:state_dim,k=1:K+1] - dxv[j,k] .== state_lin[j,k])
	#delta_uvs = @constraint(m, uv[j=1:control_dim,k=1:K+1] - duv[j,k] .== control_lin[j,k])

	# set up the objective 
	#=
	for i=1:K+1
		@constraint(m, [thr[i], uv[:,i]...] in MOI.SecondOrderCone(control_dim+1))
	end
	@constraint(m, tot_thr==sum(thr[:]))
	=#
	@objective(m, Min, -xv[mass_idx,K+1] + prob.wID*delk + prob.wDS*dels) # todo

	#vars = [nuv[j,i] for j in 1:state_dim, i in 1:K]
	#@constraint(m, [nuNorm, vars...] in MOI.SecondOrderCone(1+state_dim*(K)))

	#initial state
	@constraint(m, xv[mass_idx,1] == prob.mwet)
	@constraint(m, xv[r_idx_it,1] .== prob.rIi)
	@constraint(m, xv[v_idx_it,1] .== prob.vIi)
	@constraint(m, xv[omb_idx_it,1] .== prob.wBi)
	#@constraint(m, xv[qbi_idx_it,1] .== prob.qBIi)

	#final state
	@constraint(m, xv[r_idx_it,K+1] .== prob.rIf)
	@constraint(m, xv[v_idx_it,K+1] .== prob.vIf)
	@constraint(m, xv[omb_idx_it,K+1] .== prob.wBf)
	@constraint(m, xv[qbi_idx_it,K+1] .== prob.qBIf)
	@constraint(m, uv[2:3,K+1] .== [0,0])

	#dynamics
	dynamic_constraints = Vector{Vector{JuMP.ConstraintRef}}()
	for i=1:K
		deriv = iterDynam[i].derivative
		ep = iterDynam[i].endpoint - iterAbout[i+1].state
		push!(dynamic_constraints, @constraint(m, deriv * [xv[:,i]-iterAbout[i].state;uv[:,i]-iterAbout[i].control;uv[:,i+1]-iterAbout[i+1].control;sv-sigHat] + iterDynam[i].endpoint .== xv[:,i+1]))
	end

	#state constraints
	pointing_constraints = Vector{JuMP.ConstraintRef}()

	normed = map(lin->lin.control/norm(lin.control), iterAbout)
	mdry = prob.mdry
	for i=2:K+1
		@constraint(m, mdry .<= xv[mass_idx,i])
		#@constraint(m, [xv[2,i]/tggs, xv[3,i], xv[4,i]] in MOI.SecondOrderCone(3))
		#@constraint(m, [sqcm, 1, xv[10,i], xv[11,i]] in MOI.RotatedSecondOrderCone(4))
		#@constraint(m, [prob.omMax, xv[omb_idx_it,1]...] in MOI.SecondOrderCone(4))
	end
	for i=1:K+1
		#control constraints
		push!(pointing_constraints, @constraint(m, prob.Tmin <= dot(normed[i], uv[:,i])))
		@constraint(m, [prob.Tmax, uv[:,i]...] in MOI.SecondOrderCone(4))
		@constraint(m, [uv[2,i]/delMax, uv[:,i]...] in MOI.SecondOrderCone(4))
	end
	@constraint(m, [1/2, delk, [xv[j,i]-iterAbout[i].state[j] for j in 1:state_dim, i in 1:K+1]..., [uv[j,i]-iterAbout[i].control[j] for j in 1:control_dim, i in 1:K+1]...] in MOI.RotatedSecondOrderCone(2+state_dim*(K+1)+control_dim*(K+1)))
	#@constraint(m, [prob.ri, [uv[j,i]-iterAbout[i].control[j] for j in 1:control_dim, i in 1:K+1]...] in MOI.SecondOrderCone(1+control_dim*(K+1)))
	@constraint(m, [1/2, dels, sv-sigHat] in MOI.RotatedSecondOrderCone(3))
	return m,xv,dels,delk
	return ProblemModel(m, xv, uv, nuv, dels, dynamic_constraints, pointing_constraints)
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
			MOI.modifyconstraint!(moimod, state_base[j,i].index, MOI.EqualTo(st[j]))
		end
		for j=1:control_dim
			MOI.modifyconstraint!(moimod, control_base[j,i].index, MOI.EqualTo(ct[j]))
		end

		#pointing 
		normed = iterAbout[i].control/norm(iterAbout[i].control)
		for j=1:control_dim
			MOI.modifyconstraint!(moimod, pointing_constraints[i].index, MOI.ScalarCoefficientChange(uv[j,i].index, -normed[j]))
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
				MOI.modifyconstraint!(moimod, cstrs[j].index, MOI.ScalarCoefficientChange(vrs[vi].index, deriv[j,vi]))
			end
			MOI.modifyconstraint!(moimod, cstrs[j].index, MOI.EqualTo(-endpoint[j]))
		end
	end

	JuMP.optimize(iteration.model.socp_model)

    #=

	normed = map(lin->lin.control/norm(lin.control), iterAbout)
	consts = convert(Array{Float64,1},vcat(normed...))
	indexes = repeat(1:51,inner=3)
	vars = reshape(uv, length(uv))
	println(size(consts))
	for i=1:length(vars)
		MOI.modifyconstraint!(model, iteration.model.thrust_constraint, MOI.MultirowChange(vars[i], [indexes[i]], [consts[i]]))
	end
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