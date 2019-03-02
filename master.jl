module RocketlandDefns
	using StaticArrays
	using Interpolations
	using MathOptInterface
	abstract type AerodynamicInfo end

	struct ExoatmosphericData <: AerodynamicInfo end

	struct AtmosphericData <: AerodynamicInfo
		drag_itrp :: Interpolations.ScaledInterpolation{Float64, 2, T, U, V} where {T, U, V}
		lift_itrp :: Interpolations.ScaledInterpolation{Float64, 2, T, U, V} where {T, U, V}
		trq_itrp :: Interpolations.ScaledInterpolation{Float64, 2, T, U, V} where {T, U, V}
		force_scalar :: Float64
		length_scalar :: Float64
	end
	struct DescentProblem
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
	    sos::Float64 # speed of sound m/s

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

	    aero::AerodynamicInfo

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
	                    jB=diagm(0=>[1e-2,1e-2,1e-2]), alpha=0.01, rho=1.225,rTB=[-1e-2,0,0],rFB=[1e-2,0,0],rIi=[4.0,4.0,0.0],rIf=[0.0,0.0,0.0],
	                    vIi=[0,-2,2],vIf=[-0.1,0.0,0.0],qBIi=[1.0,0,0,0],qBIf=[1.0,0,0,0],wBi=[0.0,0.0,0.0], aero = ExoatmosphericData(),
	                    wBf=[0.0,0,0],K=50,imax=15,wNu=1e5,wID=1e-3, wDS=1e-1, wCst=10.0, wTviol=100.0, nuTol=1e-10, delTol = 1e-3, tf_guess=1.0, ri=1.0, rh0=0.0, rh1=0.25, rh2=0.90, alph=2.0, bet=3.2, sos=5.0) =
	        new(g,mdry,mwet,Tmin,Tmax,deltaMax,thetaMax,gammaGs,omMax,dpMax,jB,alpha,rho,sos,rTB,rFB,rIi,rIf,vIi,vIf,
	            qBIi,qBIf,wBi,wBf,aero,K,imax,wNu,wID,wDS,wCst,wTviol,nuTol,delTol,tf_guess,ri,rh0,rh1,rh2, alph, bet)
	end

	struct ProbInfo
	    a::Float64
	    g0::Float64
	    sos::Float64
	    jB::SArray{Tuple{3,3},Float64,2,9}
	    jBi::SArray{Tuple{3,3},Float64,2,9}
	    rTB::SArray{Tuple{3}, Float64, 1, 3}
	    rFB::SArray{Tuple{3}, Float64, 1, 3}
	    aero::AerodynamicInfo
	    ProbInfo(from::DescentProblem) = new(from.alpha, from.g, from.sos, SMatrix{3,3}(from.jB), SMatrix{3,3}(inv(from.jB)), SVector{3}(from.rTB), SVector{3}(from.rFB), from.aero)
	end

	struct LinPoint
	    state::Array{Float64, 1} # SArray{Tuple{14}, Float64, 1, 14}
	    control::Array{Float64, 1} # SArray{Tuple{3}, Float64, 1, 3}
	end

    struct LinRes
        endpoint::Array{Float64, 1}#SArray{Tuple{14},Float64,1,14}
        derivative::Array{Float64, 2}#SArray{Tuple{14,21},Float64,2,294}
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
		state_base::MOI.ConstraintIndex
		control_base::MOI.ConstraintIndex
		dynamic_constraints::Vector{MOI.ConstraintIndex}
		thrust_lb_constraint::MOI.ConstraintIndex
	end

    struct LinearCache
        cfg
        cache :: Any # don't know until inner module is generated
        probinfo :: ProbInfo
        base_dt :: Float64
        lin_mod
    end

	struct ProblemIteration
		problem::DescentProblem
		cache::LinearCache
		sigma::Float64

		about::Array{LinPoint,1}
		dynam::Array{LinRes,1}
		model::ProblemModel

		iter::Int64
		rk::Float64
		cost::Float64
	end
	export DescentProblem, ProbInfo, LinPoint, LinRes, ProblemIteration, ProblemModel, AerodynamicInfo, AtmosphericData, ExoatmosphericData, LinearCache
end
include("aerodynamics.jl")
include("symbolic_diff.jl")
include("dynamics.jl")
include("initial_solve.jl")
include("rocketland.jl")
include("sample_problems.jl")
