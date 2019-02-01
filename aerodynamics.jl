module Aerodynamics
using DataFrames
using CSV
using LinearAlgebra
using StaticArrays
using Interpolations
using SymEngine
using ..RocketlandDefns

function load_aerodata(liftdrag::String)
	lddf = CSV.read(liftdrag) #numbers in newtons
	#normalize
	lddf[:lift_p] = lddf[:lift]
	lddf[:drag_p] = lddf[:drag]
	#mach, aoa
	mach=0.0:0.025:1.5
	aoa=cosd(180):1/90:cosd(0)
	lift_itrp = scale(interpolate(reshape(lddf[:lift_p],length(aoa), length(mach)), BSpline(Cubic(Line(OnGrid())))), aoa, mach)
	drag_itrp = scale(interpolate(reshape(lddf[:drag_p],length(aoa), length(mach)), BSpline(Cubic(Line(OnGrid())))), aoa, mach)
	AtmosphericData(drag_itrp, lift_itrp, 1.0)
end

function rescale_aerodata(data::AtmosphericData, Ul::Float64, Ut::Float64, Um::Float64)
	return AtmosphericData(data.drag_itrp, data.lift_itrp, 1/(Ul * Um/Ut^2))
end

function rescale_aerodata(data::ExoatmosphericData, Ul::Float64, Ut::Float64, Um::Float64)
	return data
end

function aero_force(data::AtmosphericData, bv::AbstractArray{T}, vel::AbstractArray{T}, spds::Float64)::Vector{T} where T<:Number
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

function aero_force(data::AtmosphericData, bv::AbstractArray{T}, vel::AbstractArray{T}, spds::Float64)::Vector{T} where T<:Basic
	dp = sum(bv .* vel)
	cos_aoa = dp #/sqrt(sum(bv .^ 2))
	mach = sqrt(sum(vel .^ 2))/spds

	drag = SymFunction("drag")(cos_aoa,mach,symbols("pinfo"))*data.scalar
	lift = SymFunction("lift")(cos_aoa,mach,symbols("pinfo"))*data.scalar
	liftd = cross(cross(bv, vel),vel)
	liftd_norm = sqrt(sum(liftd .^ 2))#sqrt(sum(vel .^ 2) * sum(bv .^ 2) * (sum(vel .^ 2) - cos_aoa^2))
	liftd = (liftd)/liftd_norm
	dragf = drag*vel/sqrt(sum(vel .^ 2))
	liftf = lift*(SymFunction("ifnz").(liftd_norm, liftd))
	return dragf + liftf
end

function aero_force(data::ExoatmosphericData, bv::AbstractArray{T} where T, vel::AbstractArray{T} where T, spds::Float64)
	return [0.0,0.0,0.0]
end

function direct_drag(data::AtmosphericData, cos_aoa, mach)
	return data.drag_itrp(cos_aoa, mach)
end

function direct_lift(data::AtmosphericData, cos_aoa, mach)
	return data.lift_itrp(cos_aoa, mach)
end

function direct_drag(data::ExoatmosphericData, cos_aoa, mach)
	return 0.0
end

function direct_lift(data::ExoatmosphericData, cos_aoa, mach)
	return 0.0
end


end