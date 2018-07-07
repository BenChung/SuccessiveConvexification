using Interpolations
using kRPC
using CSV
conn = kRPCConnect("Aerodata")
using kRPC.Remote.SpaceCenter
totable = 1
outp_a = 1
outp_b = 1
#try 
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

equivalent_vel = norm([Position(vessel, body_frame)...]) * RotationalSpeed(body)
base_vector = Float64[0, 0, -equivalent_vel]
direction = Float64[-1,0,0]
speed_of_sound = SpeedOfSound(flight)

# v.v runs from 0 to max mach^2
# b.v runs from -sqrt(v.v) to sqrt(v.v) NOW NORMALIZED

function compute_aero_forces(mach, aoa)
	velocity = mach*speed_of_sound
	velocity_vec = broadcast(*, Float64[-cos(aoa),sin(aoa),0], velocity)
	return kRPC.Remote.SpaceCenter.Delayed.SimulateAerodynamicForceAt(guidance_flight, body, (0.0,0.0,0.0), ((velocity_vec + base_vector)...)), velocity_vec
end

cforce = kRPC.kPC[]
dp = Float64[]
for aoa=-10:0.1:10
	cll,vec = compute_aero_forces(0.5, deg2rad(aoa))
	push!(cforce, cll)
	b = [cosd(aoa), sind(aoa), 0.0]
	v = [0.5,0.0,0.0]
	n = norm(cross(cross(b,v),b))
	println(n-sqrt(1-(dot(b,v)/norm(v))^2))
	push!(dp, norm(cross(b,v))/norm(v))
end
res = (x->x[2]).(kRPC.SendMessage(conn, cforce))

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


finally
kRPCDisconnect(conn)
end
=#