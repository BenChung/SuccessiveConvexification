using Interpolations
using kRPC
using CSV
conn = kRPCConnect("Aerodata")
using kRPC.Remote.SpaceCenter
totable = 1
outp_a = 1
outp_b = 1
try 
vessel = ActiveVessel()
flight = Flight(vessel)
orbit = Orbit(vessel)
body = Body(orbit)
body_frame = ReferenceFrame(body)

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

guidance_flight = Flight(vessel, referenceFrame=guidance_frame)

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

function compute_aero_forces_tform(bv,vv)
	if (vv>0)
		alpha = acos(bv)
	else
		alpha = 0 #zero velocity anyways
	end
	mach = sqrt(vv)
	return compute_aero_forces(mach, alpha)
end



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