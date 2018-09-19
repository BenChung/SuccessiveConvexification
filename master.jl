module RocketlandDefns
	using StaticArrays
	using MathOptInterface

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
	    dpMax::Float64 # max dynamic pressure Pa
	    jB::Array{Float64,2}
	    alpha::Float64
	    rho::Float64 # atmospheric density kg/m^3

	    rTB::Array{Float64,1}
	    rFB::Array{Float64,1}

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
	    wCst::Float64
	    wTviol::Float64
	    nuTol::Float64
	    delTol::Float64
	    tf_guess::Float64

	    ri::Float64
	    rh0::Float64
	    rh1::Float64
	    rh2::Float64
	    alph::Float64
	    bet::Float64

	    DescentProblem(;g=1.0,mdry=1.0,mwet=2.0,Tmin=0.3,Tmax=5.0,deltaMax=20.0,thetaMax=90.0,gammaGs=20.0,dpMax=50000,omMax=60.0,
	                    jB=diagm([1e-2,1e-2,1e-2]), alpha=0.01, rho=1.225,rTB=[-1e-2,0,0],rFB=[1e-2,0,0],rIi=[4.0,4.0,0.0],rIf=[0.0,0.0,0.0],
	                    vIi=[0,-2,2],vIf=[-0.1,0.0,0.0],qBIi=[1.0,0,0,0],qBIf=[1.0,0,0,0],wBi=[0.0,0.0,0.0],
	                    wBf=[0.0,0,0],K=50,imax=15,wNu=1e5,wID=1e-3, wDS=1e-1, wCst=10.0, wTviol=100.0, nuTol=1e-10, delTol = 1e-3, tf_guess=1.0, ri=1.0, rh0=0.0, rh1=0.25, rh2=0.90, alph=2.0, bet=3.2) =
	        new(g,mdry,mwet,Tmin,Tmax,deltaMax,thetaMax,gammaGs,omMax,dpMax,jB,alpha,rho,rTB,rFB,rIi,rIf,vIi,vIf,
	            qBIi,qBIf,wBi,wBf,K,imax,wNu,wID,wDS,wCst,wTviol,nuTol,delTol,tf_guess,ri,rh0,rh1,rh2, alph, bet)
	end

	function normalize_problem(dp::DescentProblem)::DescentProblem
		Ul = maximum(dp.rIi)
		Ut = dp.tf_guess
		Um = dp.mwet
		return DescentProblem(
			g=dp.g/(Ul/Ut^2), mdry=dp.mdry/Um, mwet=dp.mwet/Um,
			Tmin = dp.Tmin/(Um * Ul/Ut^2), Tmax = dp.Tmax/(Um * Ul/Ut^2),
			omMax = dp.omMax/Ut, jB = broadcast(*, dp.jB, 1/(Um * Ul^2)),
			rTB = broadcast(*, dp.rTB, 1/Ul), rIi = broadcast(*, dp.rIi, 1/Ul),
			rIf = broadcast(*, dp.rIf, 1/Ul), vIi = broadcast(*, dp.vIi, 1/(Ul/Ut)),
			vIf = broadcast(*, dp.vIi, 1/(Ul/Ut)), qBIf = dp.qBIf, qBIi = dp.qBIi,
			wBi = dp.wBi, wBf = dp.wBf, rFB = broadcast(*, dp.rFB, 1/Ut),
			deltaMax = dp.deltaMax, thetaMax = dp.thetaMax, gammaGs = dp.gammaGs,
			alpha = dp.alpha/(Ut^2/Ul), K=dp.K, imax = dp.imax, wNu = dp.wNu, wID = dp.wID,
			wDS = dp.wDS, wCst = dp.wCst, wTviol = dp.wTviol, delTol = dp.delTol,
			tf_guess = dp.tf_guess/Ut, ri = dp.ri, rh0 = dp.rh0, rh1 = dp.rh1,
			rh2 = dp.rh2, alph = dp.alph, bet = dp.bet, dpMax=dp.dpMax/(Um/(Ul*Ut^2)), rho = dp.rho/(Um/Ul^3))
	end

	base_prob = DescentProblem(g=9.82, mdry=66018, mwet=92960, Tmin=0.4*1000000, Tmax=1000000, jB=diagm([1.65e5,8.77e6,8.773e6]), 
										  alpha=0.0003449, rTB=[-9.78571,0,0], rIi = [1000.0,1000.0,100.0], rIf=[0.0,0.0,0.0], vIi = [-100.0,0,0])
	base_prob_scaled = normalize_problem(base_prob)

	struct ProbInfo
	    a::Float64
	    g0::Float64
	    jB::SArray{Tuple{3,3},Float64,2,9}
	    jBi::SArray{Tuple{3,3},Float64,2,9}
	    rTB::SArray{Tuple{3}, Float64, 1, 3}
	    ProbInfo(from::DescentProblem) = new(from.alpha, from.g, SMatrix{3,3}(from.jB), SMatrix{3,3}(inv(from.jB)), SVector{3}(from.rTB))
	end

	struct LinPoint
	    state::SArray{Tuple{14}, Float64, 1, 14}
	    control::SArray{Tuple{3}, Float64, 1, 3}
	end

    struct LinRes
        endpoint::SArray{Tuple{14},Float64,1,14}
        derivative::SArray{Tuple{14,21},Float64,2,294}
    end

    const MOI=MathOptInterface
	struct ProblemModel
		socp_model::MOI.ModelLike
		xv::Array{MOI.VariableIndex,2}
		uv::Array{MOI.VariableIndex,2}
		dxv::Array{MOI.VariableIndex,2}
		duv::Array{MOI.VariableIndex,2}
		dsv::MOI.VariableIndex
		nuv::Array{MOI.VariableIndex,2}
		tnv::Array{MOI.VariableIndex,1}
		anv::Array{MOI.VariableIndex,1}
		rkv::MOI.VariableIndex
		state_base::MOI.ConstraintIndex
		control_base::MOI.ConstraintIndex
		dynamic_constraints::Vector{MOI.ConstraintIndex}
		pointing_constraints::MOI.ConstraintIndex
		aoa_constraint::MOI.ConstraintIndex
		trust_region::MOI.ConstraintIndex
	end

	struct ProblemIteration
		problem::DescentProblem
		sigma::Float64

		about::Array{LinPoint,1}
		dynam::Array{LinRes,1}
		model::ProblemModel

		iter::Int64
		rk::Float64
		cost::Float64
	end
	export DescentProblem, ProbInfo, LinPoint, LinRes, ProblemIteration, ProblemModel
end
include("dynamics.jl")
include("initial_solve.jl")
include("rocketland.jl")