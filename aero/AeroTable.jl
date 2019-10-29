using Interpolations
using kRPC
using CSV
using LinearAlgebra
using DataFrames
conn = kRPCConnect("Aerodata")
using kRPC.Remote.SpaceCenter
totable = 1
outp_a = 1
outp_b = 1

vessel = ActiveVessel()
flight = Flight(vessel)
orbit = Orbit(vessel)
body = Body(orbit)
body_frame = ReferenceFrame(body)
control = Control(vessel)

function compute_reference_frame(body, lat, lon, alt::Float64)
    landing_position = SurfacePosition(body, lat, lon, body_frame)
    q_long = (
        0.0,
        sin(-lon * 0.5 * pi / 180),
        0.0,
        cos(-lon * 0.5 * pi / 180))
    q_lat = (
        0.0,
        0.0,
        sin(lat * 0.5 * pi / 180),
        cos(lat * 0.5 * pi / 180))
    cr1 = CreateRelative(body_frame, position=landing_position, rotation=q_long)
    cr2 = CreateRelative(cr1,rotation=q_lat)
    return CreateRelative(cr2,position=(alt,0.0,0.0))
end
guidance_frame = compute_reference_frame(body, Latitude(flight), Longitude(flight), 0.0)

guidance_flight = Flight(vessel)#, referenceFrame=guidance_frame)
direction = Float64[-1,0,0]
speed_of_sound = SpeedOfSound(flight)

function compute_aero_forces(mach, cos_aoa)
	velocity = mach*speed_of_sound
	velocity_vec = broadcast(*, Float64[cos_aoa,sqrt(1-cos_aoa^2),0], velocity)
	return kRPC.Remote.SpaceCenter.Delayed.SimulateAerodynamicForceAt(guidance_flight, body, (0.0,0.0,0.0), ((velocity_vec)...,)),
		   kRPC.Remote.SpaceCenter.Delayed.SimulateAerodynamicTorqueAt(guidance_flight, body, (0.0,0.0,0.0), ((velocity_vec)...,)), velocity_vec
end

function sweep_aero()
	cforce = kRPC.kPC[]
	ctorque = kRPC.kPC[]
	drg = Vector{Float64}[]
	lft = Vector{Float64}[]
	trq = Vector{Float64}[]
	aoal = Float64[]
	machl = Float64[]
	for mach=0.0:0.025:1.5
		for cos_aoa=cosd(180):1/90:cosd(0)
			cll,ctr,bv = compute_aero_forces(mach, cos_aoa)
			push!(cforce, cll)
			push!(ctorque, ctr)
			normbv = bv/norm(bv)
			push!(drg, normbv)
			torqd = cross(normbv, [1.0,0.0,0.0])
			liftd = cross(torqd,normbv)
			liftd = liftd/norm(liftd)
			push!(lft, liftd)
			push!(trq, torqd/norm(torqd))
			push!(aoal, cos_aoa)
			push!(machl, mach)
		end
	end
	res = kRPC.SendMessage(conn, cforce)
	tres = kRPC.SendMessage(conn, ctorque)
	hcat(([aoal[i], machl[i], 
			if !isnan(dot(res[i], drg[i])) dot(res[i], drg[i]) else 0.0 end, 
			if !isnan(dot(res[i], lft[i])) dot(res[i], lft[i]) else 0.0 end, 
			if !isnan(dot(tres[i], trq[i])) dot(tres[i], trq[i]) else 0.0 end] for i=1:length(res))...)
end

function compute_aero_table()
	aero_force = sweep_aero()'
	aero_df = DataFrame(aoa=aero_force[:,1], mach=aero_force[:,2], drag=aero_force[:,3], lift=aero_force[:,4], torque=aero_force[:,5])
	CSV.write("lift_drag.csv", aero_df);
end

pts = Parts(vessel)
finsOn = WithTag(pts, "nosefins2")
fin1 = Modules(finsOn[1])[1]
fin2 = Modules(finsOn[2])[1]
finsOff = WithTag(pts, "nosefins")
fin3 = Modules(finsOff[1])[1]
fin4 = Modules(finsOff[2])[1]

function compute_fin_force_table()
	SetFieldFloat.([fin3,fin4], "Flp/Splr Dflct", convert(Float32, 0))
	SetFieldFloat.([fin1,fin2], "Flp/Splr Dflct", convert(Float32, 0))
	sleep(4)

	function aero_profile()
		return map(x -> [x...], kRPC.SendMessage(conn, [begin frcc,trqc,vv = compute_aero_forces(mach, 1.0); frcc end for mach=0.01:0.025:1.5]))
	end
	bl = aero_profile()
	forces = Vector{Vector{Float64}}[]
	for ang=0.0:0.1:90.0
		SetFieldFloat.([fin1,fin2], "Flp/Splr Dflct", convert(Float32, ang))
		sleep(0.1)
		push!(forces, aero_profile())
	end
	aerod = hcat(map((force,aoa) -> vcat(hcat((force .- bl)...), transpose(collect(0.01:0.025:1.5)), fill(aoa, 1, length(bl))), forces, 0.0:0.1:90.0)...)
	df = DataFrame(lift=aerod[2,:], drag=aerod[1,:], mach=aerod[4,:], aoa=aerod[5,:])
	CSV.write("fin.csv", df);
end

#=

function compute_aero_forces_tform(bv,vv)
	if (vv>0)
		alpha = acos(bv)
	else
		alpha = 0 #zero velocity anyways
	end
	mach = sqrt(vv)
	return compute_aero_forces(mach, alpha)
end

ctrls = kRPC.Remote.FerramControlSurface.FlightControls(vessel)
map(x->kRPC.Remote.FerramControlSurface.ControlMaxDeflection!(x, 80.0f0), ctrls)


function compute_control_force_defl(mach, aoamin, aoamax, minangle, maxangle, func)
	kRPC.Remote.FerramControlSurface.ControlTimeConstant!(0.00000001)
	func(control, minangle)
	sleep(0.1)

	forces = []
	for aoa in aoamin:pi/64:aoamax
		func(control, minangle)
		sleep(0.2)
		basf = kRPC.SendMessage(conn, compute_aero_forces(mach, aoa)[1])
		for angle in minangle:0.1f0:maxangle
			func(control, angle)
			println(angle)
			sleep(0.1)
			frce = [[(basf .- kRPC.SendMessage(conn, compute_aero_forces(mach, aoa)[1]))...] for i=1:10]
			mfrce = mean(frce)
			unshift!(mfrce, aoa)
			push!(forces, mfrce)
		end
	end
	func(control, 0.0f0)
	sleep(0.1)
	return hcat(forces...)
end

function compute_control_force_drag(mach, angle, delta)
	start_a = angle - delta/2
	end_a = angle + delta/2
	stepsize = delta/40
	force = []
	kRPC.Remote.FerramControlSurface.ControlTimeConstant!(0.00000001)
	Pitch!(control, convert(Float32, angle/deg2rad(80.0)))
	sleep(0.3)
	basef = kRPC.SendMessage(conn, compute_aero_forces(mach, pi+angle)[1])
	Pitch!(control, convert(Float32, start_a/deg2rad(80.0)))
	sleep(0.3)
	for ang in start_a:stepsize:end_a
		Pitch!(control, convert(Float32, ang/deg2rad(80.0)))
		sleep(0.05)
		push!(force, [(kRPC.SendMessage(conn, compute_aero_forces(mach, pi+angle)[1]) .- basef)..., ang])
	end
	return basef,force
end

function find_apex(mach, angle, delta)
	start_a = angle - delta/2
	end_a = angle + delta/2
	stepsize = delta/10
	angles = Any[]
	for ang in start_a:stepsize:end_a
		bf,fr = compute_control_force_drag(mach, ang, delta)
		fr = hcat(fr...)
		sz = size(fr)[2]
		which_angle = indmin(abs.((fr[1,1:sz-1] - fr[1,2:sz])./(fr[2,1:sz-1] - fr[2,2:sz])))
		push!(angles, [bf,fr])
	end
	return angles
end

function fit_parabola(t,b)
	inp = hcat(t.^2,t.^4)
	println(inp, " ", b)
	return inp\b
end

#=
bv_step = 0.005
vv_step = 0.1
max_mach = 2
reqs = [(bv,vv,compute_aero_forces_tform(bv,vv))
             for vv in 0:vv_step:max_mach^2
             for bv in 1:-bv_step:-1]
tosend = map(x->x[3][1], reqs)
res = kRPC.SendMessage(conn, tosend)

density = AtmosphereDensity(flight)
function decompose(aero, velocity, bv, vv)
	iv = velocity
	ia = aero
	velocity = velocity[1:2] # hey this is really a 2d problem
	aero = aero[1:2]/speed_of_sound
	velocity_dir = velocity/norm(velocity)
	drag = dot(aero,velocity_dir)
	lift = dot([-velocity_dir[2],velocity_dir[1]], aero - drag*velocity_dir)

	bx = bv
	by = sqrt(1-bx^2)

	if abs(by) < 0.0001
		res= (0.0, drag/sqrt(vv))
	else
		res= (lift/by, (drag-bx*lift/by)/sqrt(vv))
	end
	if isnan(res[1]) res = (0.0, res[2]) end
	if isnan(res[2]) res = (res[1], 0.0) end
	return res
end
totable = map((inp,res)->(inp[1],inp[2])=>decompose([res...], inp[3][2], inp[1], inp[2]), reqs, res)

bvs = unique(map(x->x[1][1], totable))
machsq = unique(map(x->x[1][2], totable))
map_bv = Dict([bvs[i] => i for i = 1:length(bvs)])
map_machsq = Dict([machsq[i] => i for i = 1:length(machsq)])
outp_a = Array{Float64,2}(length(bvs), length(machsq))
outp_b = Array{Float64,2}(length(bvs), length(machsq))
for result in totable
	outp_a[map_bv[result[1][1]], map_machsq[result[1][2]]] = result[2][1]
	outp_b[map_bv[result[1][1]], map_machsq[result[1][2]]] = result[2][2]
end
=#
=#
