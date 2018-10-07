module Aerodynamics
using DataFrames
using CSV
using LinearAlgebra
using StaticArrays
using Interpolations
using ..RocketlandDefns

function load_aerodata(liftdrag::String)
	lddf = CSV.read(liftdrag) #numbers in newtons
	#normalize
	lddf[:lift_p] = lddf[:lift]
	lddf[:drag_p] = lddf[:drag]
	#mach, aoa
	mach=0.0:0.025:1.5
	aoa=cosd(180):1/90:cosd(0)
	lift_itrp = scale(interpolate(reshape(lddf[:lift_p],length(aoa), length(mach)), BSpline(Cubic(Line())), OnGrid()), aoa, mach)
	drag_itrp = scale(interpolate(reshape(lddf[:drag_p],length(aoa), length(mach)), BSpline(Cubic(Line())), OnGrid()), aoa, mach)
	AtmosphericData(drag_itrp, lift_itrp, 1.0)
end

function rescale_aerodata(data::AtmosphericData, Ul::Float64, Ut::Float64, Um::Float64)
	return AtmosphericData(data.drag_itrp, data.lift_itrp, 1/(Ul * Um/Ut^2))
end

function rescale_aerodata(data::ExoatmosphericData, Ul::Float64, Ut::Float64, Um::Float64)
	return data
end

function aero_force(data::AtmosphericData, bv::SArray{Tuple{3}, T, 1, 3}, vel::SArray{Tuple{3}, T, 1, 3}, spds::Float64)::Vector{T} where T
	dp = dot(bv,vel)/norm(vel)
	cos_aoa = clamp(dp/norm(bv), -1.0,1.0)
	mach = norm(vel)/spds
	if abs(dp) >= 0.95
		drag = data.drag_itrp(cos_aoa,mach)::T*data.scalar
		dragf = drag*vel/norm(vel)
		return dragf
	else
		drag = data.drag_itrp(cos_aoa,mach)::T*data.scalar
		lift = data.lift_itrp(cos_aoa,mach)::T*data.scalar
		liftd = cross(cross(bv, vel),vel)
		liftd = liftd/norm(liftd)
		dragf = drag*vel/norm(vel)
		liftf = lift*liftd
		return dragf + liftf
	end
end

function aero_force(data::ExoatmosphericData, bv::SArray{Tuple{3}, T, 1, 3} where T, vel::SArray{Tuple{3}, T, 1, 3} where T, spds::Float64)
	return [0.0,0.0,0.0]
end

end