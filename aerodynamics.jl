module Aerodynamics
using DataFrames
using CSV
using Interpolations

struct AerodynamicData
end

function load_aerodata(liftdrag::String, Ul::Float64, Ut::Float64, Um::Float64)
	lddf = CSV.read(liftdrag) #numbers in newtons
	#normalize
	lddf[:lift_p] = lddf[:lift]/(Ul * Um/Ut^2)
	lddf[:drag_p] = lddf[:drag]/(Ul * Um/Ut^2)
	#mach, aoa
	mach=0.0:0.1:1.0
	aoa=0:1:45
	lift_itrp = scale(interpolate(reshape(lddf[:lift_p],length(aoa), length(mach)), BSpline(Cubic(Line())), OnGrid()), aoa, mach)
	drag_itrp = scale(interpolate(reshape(lddf[:drag_p],length(aoa), length(mach)), BSpline(Cubic(Line())), OnGrid()), aoa, mach)
	lddf, lift_itrp
end

end