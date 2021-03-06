module Aerodynamics
using DataFrames
using CSV
using LinearAlgebra
using StaticArrays
using Interpolations
using SymEngine
using ..RocketlandDefns
import StaticArrays

function load_aerodata(liftdrag::String, finforce::Union{String, Nothing}=nothing)
	lddf = CSV.read(liftdrag) #numbers in newtons
	#normalize
	lddf[:lift_p] = lddf[:lift]
	lddf[:drag_p] = lddf[:drag]
	#mach, aoa
	mach=0.0:0.025:1.5
	aoa=cosd(180):1/90:cosd(0)
	lift_itrp = extrapolate(scale(interpolate(reshape(lddf[:lift_p],length(aoa), length(mach)), BSpline(Cubic(Line(OnGrid())))), aoa, mach), Flat())
	drag_itrp = extrapolate(scale(interpolate(reshape(lddf[:drag_p],length(aoa), length(mach)), BSpline(Cubic(Line(OnGrid())))), aoa, mach), Flat())
	trq_itrp = extrapolate(scale(interpolate(reshape(lddf[:torque],length(aoa), length(mach)), BSpline(Cubic(Line(OnGrid())))), aoa, mach), Flat())

	if !isnothing(finforce)
		ffdf = CSV.read(finforce)
		
	end
	AtmosphericData(drag_itrp, lift_itrp, trq_itrp, 1.0, 1.0)
end

function rescale_aerodata(data::AtmosphericData, Ul::Float64, Ut::Float64, Um::Float64)
	return AtmosphericData(data.drag_itrp, data.lift_itrp, data.trq_itrp, 1/(Ul * Um/Ut^2), 1/Ul)
end

function rescale_aerodata(data::ExoatmosphericData, Ul::Float64, Ut::Float64, Um::Float64)
	return data
end

function aero_force(data::AtmosphericData, bv::AbstractArray{T}, vel::AbstractArray{T}, spds::Float64)::Tuple{Vector{T}, Vector{T}} where T<:Number
	dp = dot(bv,vel)/norm(vel)
	cos_aoa = clamp(dp/norm(bv), -1.0,1.0)
	mach = norm(vel)/spds
	if abs(dp) >= 0.95
		drag = data.drag_itrp(cos_aoa,mach)::T*data.force_scalar
		dragf = drag*vel/norm(vel)
		return dragf, zeros(T, 3)
	else
		drag = data.drag_itrp(cos_aoa,mach)::T*data.force_scalar
		lift = data.lift_itrp(cos_aoa,mach)::T*data.force_scalar
		trq = data.trq_itrp(cos_aoa,mach)::T*data.length_scalar*data.force_scalar
		trqd = cross(vel, bv)
		liftd = cross(-trqd,vel)
		liftd = liftd/norm(liftd)
		trqd = trqd/norm(trqd)
		dragf = drag*vel/norm(vel)
		liftf = lift*liftd
		return dragf + liftf, trqd*trq
	end
end

function aero_force(data::AtmosphericData, bv::AbstractArray{T}, vel::AbstractArray{T}, spds::Float64)::Tuple{Vector{T}, Vector{T}} where T<:Basic
	dp = sum(bv .* vel)
	cos_aoa = dp #/sqrt(sum(bv .^ 2))
	mach = sqrt(sum(vel .^ 2))/spds

	drag = SymFunction("drag")(cos_aoa,mach,symbols("pinfo"))*data.force_scalar
	lift = SymFunction("lift")(cos_aoa,mach,symbols("pinfo"))*data.force_scalar
	trq = SymFunction("trq")(cos_aoa,mach,symbols("pinfo"))*data.length_scalar*data.force_scalar
	trqd = cross(vel, bv)
	liftd = cross(-trqd,vel)
	liftd_norm = sqrt(sum(liftd .^ 2))#sqrt(sum(vel .^ 2) * sum(bv .^ 2) * (sum(vel .^ 2) - cos_aoa^2))
	drag_norm = sqrt(sum(vel .^ 2))
	liftd = (liftd)/liftd_norm
	liftf = lift*(SymFunction("ifnz").(liftd_norm, liftd))
	dragf = SymFunction("ifnz").(drag_norm, drag*vel/drag_norm)
	total_force = dragf + liftf
	return SymFunction("ifnz").(total_force, total_force), trq*(SymFunction("ifnz").(trqd/sqrt(sum(trqd .^ 2)), trqd))
end

function aero_force(data::ExoatmosphericData, bv::AbstractArray{T} where T, vel::AbstractArray{T} where T, spds::Float64)
	return [0.0,0.0,0.0]
end

function direct_drag(data::AtmosphericData, cos_aoa, mach)
	return data.drag_itrp(cos_aoa, mach)
end

function direct_drag(::Type{Val{:jac}}, data::AtmosphericData, cos_aoa, mach)
	return Interpolations.gradient(data.drag_itrp, cos_aoa, mach)::StaticArrays.SArray{Tuple{2},Float64,1,2}
end

function direct_lift(data::AtmosphericData, cos_aoa, mach)
	return data.lift_itrp(cos_aoa, mach)
end

function direct_lift(::Type{Val{:jac}}, data::AtmosphericData, cos_aoa, mach)
	return Interpolations.gradient(data.lift_itrp, cos_aoa, mach)::StaticArrays.SArray{Tuple{2},Float64,1,2}
end

function direct_trq(data::AtmosphericData, cos_aoa, mach)
	return data.trq_itrp(cos_aoa, mach)
end

function direct_trq(::Type{Val{:jac}}, data::AtmosphericData, cos_aoa, mach)
	return Interpolations.gradient(data.trq_itrp, cos_aoa, mach)::StaticArrays.SArray{Tuple{2},Float64,1,2}
end

function direct_drag(data::ExoatmosphericData, cos_aoa, mach)
	return 0.0
end

function direct_lift(data::ExoatmosphericData, cos_aoa, mach)
	return 0.0
end

function direct_trq(data::ExoatmosphericData, cos_aoa, mach)
	return 0.0
end

function direct_drag(::Type{Val{:jac}}, data::ExoatmosphericData, cos_aoa, mach)
	return StaticArrays.SArray{Tuple{2}}(0.0,0.0)
end

function direct_lift(::Type{Val{:jac}}, data::ExoatmosphericData, cos_aoa, mach)
	return StaticArrays.SArray{Tuple{2}}(0.0,0.0)
end

function direct_trq(::Type{Val{:jac}}, data::ExoatmosphericData, cos_aoa, mach)
	return StaticArrays.SArray{Tuple{2}}(0.0,0.0)
end

export direct_trq, direct_lift, direct_drag
end