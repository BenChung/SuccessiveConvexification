module RocketlandDefns
	using StaticArrays
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

		DescentProblem(;g=1.0,mdry=1.0,mwet=2.0,Tmin=0.3,Tmax=5.0,deltaMax=20.0,thetaMax=90.0,gammaGs=20.0,omMax=60.0,
						jB=diagm([1e-2,1e-2,1e-2]), alpha=0.01,rTB=[-1e-2,0,0],rIi=[4.0,4.0,0.0],rIf=[0.0,0.0,0.0],
						vIi=[0,-1,-1],vIf=[-0.1,0.0,0.0],qBIi=[1.0,0,0,0],qBIf=[1.0,0,0,0],wBi=[0.0,0.0,0.0],
						wBf=[0.0,0,0],K=50,imax=15,wNu=1e5,wID=1e-3, wDS=1e-1, nuTol=1e-10, delTol = 1e-3, tf_guess=5.0) =
			new(g,mdry,mwet,Tmin,Tmax,deltaMax,thetaMax,gammaGs,omMax,jB,alpha,rTB,rIi,rIf,vIi,vIf,
				qBIi,qBIf,wBi,wBf,K,imax,wNu,wID,wDS,nuTol,delTol,tf_guess)
	end

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
	export DescentProblem, ProbInfo, LinPoint, LinRes
end
include("dynamics.jl")
include("rocketland.jl")