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
	    jB::Array{Float64,2}
	    alpha::Float64

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
	    nuTol::Float64
	    delTol::Float64
	    tf_guess::Float64

	    ri::Float64
	    rh0::Float64
	    rh1::Float64
	    rh2::Float64
	    alph::Float64
	    bet::Float64

	    DescentProblem(;g=1.0,mdry=1.0,mwet=2.0,Tmin=2.0,Tmax=5.0,deltaMax=20.0,thetaMax=90.0,gammaGs=20.0,omMax=60.0,
	                    jB=diagm([1e-2,1e-2,1e-2]), alpha=0.01,rTB=[-1e-2,0,0],rFB=[1e-2,0,0],rIi=[4.0,4.0,0.0],rIf=[0.0,0.0,0.0],
	                    vIi=[0,-2,2],vIf=[-0.1,0.0,0.0],qBIi=[1.0,0,0,0],qBIf=[1.0,0,0,0],wBi=[0.0,0.0,0.0],
	                    wBf=[0.0,0,0],K=50,imax=15,wNu=1e5,wID=1e-3, wDS=1e-1, nuTol=1e-10, delTol = 1e-3, tf_guess=5.0, ri=1.0, rh0=0.0, rh1=0.25, rh2=0.90, alph=2.0, bet=3.2) =
	        new(g,mdry,mwet,Tmin,Tmax,deltaMax,thetaMax,gammaGs,omMax,jB,alpha,rTB,rFB,rIi,rIf,vIi,vIf,
	            qBIi,qBIf,wBi,wBf,K,imax,wNu,wID,wDS,nuTol,delTol,tf_guess,ri,rh0,rh1,rh2, alph, bet)
	end
	base_prob = DescentProblem(g=9.82, mdry=66018, mwet=92960, Tmin=0.4*1000000, Tmax=1000000, jB=diagm([1.65e5,8.77e6,8.773e6]), 
										  alpha=0.0003449, rTB=[-9.78571,0,0], rIi = [1000.0,1000.0,0.0], rIf=[0.0,0.0,0.0], vIi = [-50.0,0,0])
	base_prob_scaled = DescentProblem(g=0.00982, mdry=0.710, mwet=1.0, Tmin=0.04302925989672978, Tmax=0.10757314974182443, jB=diagm([1.7749569707401032e-06, 9.434165232358004e-05, 9.437392426850258e-05]), 
										  alpha=0.3449, rTB=[-0.00978571,0,0], rIi = [1.0,1.0,0.0], rIf=[0.0,0.0,0.0], vIi = [-0.05,0,0])

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
		rkv::MOI.VariableIndex
		state_base::MOI.ConstraintIndex
		control_base::MOI.ConstraintIndex
		dynamic_constraints::Vector{MOI.ConstraintIndex}
		pointing_constraints::MOI.ConstraintIndex
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