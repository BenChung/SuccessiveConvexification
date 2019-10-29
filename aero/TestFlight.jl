using kRPC
using CSV
using LinearAlgebra
using DataFrames
using Plots
pyplot()

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

rf = compute_reference_frame(body, -0.09720792138114814, -74.55767985150328, 0.0)
gfl = Flight(vessel, referenceFrame=body_frame)

cforce = Vector{Float64}[]
drg = Vector{Float64}[]
lft = Vector{Float64}[]
aoal = Float64[]
machl = Float64[]
p=plot([],[], xlims = (0,180), ylims = (0,1), legend=false, linetype=:scatter)
iter = [0]
laoa = [-10000.0]
lmach = [-10000.0]

rtb = RotationalSpeed(body)

drag_vis = kRPC.Remote.Drawing.AddLine((0.0,0.0,0.0),(0.0,0.0,0.0),body_frame)
lift_vis = kRPC.Remote.Drawing.AddLine((0.0,0.0,0.0),(0.0,0.0,0.0),body_frame)
aerf_vis = kRPC.Remote.Drawing.AddLine((0.0,0.0,0.0),(0.0,0.0,0.0),body_frame)
kRPC.Remote.Drawing.Color!(lift_vis, (1.0,0.0,0.0))
kRPC.Remote.Drawing.Color!(aerf_vis, (0.0,1.0,0.0))
global lvel = Vector{Float64}[[0.0,0.0,0.0]]
global lut = -10000.0
add_multiple_streams([kRPC.Remote.SpaceCenter.Delayed.AerodynamicForce(gfl),
	kRPC.Remote.SpaceCenter.Delayed.Direction(gfl),
	kRPC.Remote.SpaceCenter.Delayed.Velocity(gfl),
	kRPC.Remote.SpaceCenter.Delayed.Position(vessel, body_frame),
	kRPC.Remote.SpaceCenter.Delayed.AtmosphereDensity(gfl),
	kRPC.Remote.SpaceCenter.Delayed.SpeedOfSound(gfl),
	kRPC.Remote.SpaceCenter.Delayed.UT()], function (aerof,direction,velocity,posn,density,spos,iut)
	sos = Float64(spos)
	ifrc,dir,vel,pos = convert(Vector{Float64}, -[aerof...]), convert(Vector{Float64}, -[direction...]), convert(Vector{Float64}, [velocity...]), convert(Vector{Float64}, [posn...])
	aoa = acosd(dot(dir,vel)/norm(vel))
	mach = norm(vel)/sos
	if aoa == laoa[1] && mach == lmach[1]
		return
	end
	accel = [0.0,0.0,0.0]
	if lut > 0
		dt = iut - lut
		accel = (vel-lvel[1])/dt
	end
	laoa[1] = aoa
	lmach[1] = mach
	global lvel[1] = vel
	global lut = iut

	normvel = vel/norm(vel)
	push!(drg, normvel)
	lftv = cross(cross(dir, normvel),normvel)
	lftv = lftv/norm(lftv)
	push!(cforce, ifrc/density)
	push!(lft, lftv)
	push!(aoal, aoa)
	push!(machl, mach)
	iter[1] = iter[1] + 1
	if iter[1] % 10 == 0
		equivalent_vel = norm(pos) * rsb
		base_vector = Float64[0, 0, -equivalent_vel]
		sfrc = Float64[SimulateAerodynamicForceAt(gfl, body, posn, velocity)...]
		lneup = [
		 kRPC.Remote.Drawing.Delayed.Start!(drag_vis, (pos...,)),
		 kRPC.Remote.Drawing.Delayed.End!(drag_vis, ((pos + sfrc/norm(sfrc)*20)...,)),
		 kRPC.Remote.Drawing.Delayed.Start!(lift_vis, (pos...,)),
		 kRPC.Remote.Drawing.Delayed.End!(lift_vis, ((pos + accel/norm(accel)*20)...,)),
		 kRPC.Remote.Drawing.Delayed.Start!(aerf_vis, (pos...,)),
		 kRPC.Remote.Drawing.Delayed.End!(aerf_vis, ((pos + ifrc/norm(ifrc)*20) ...,))]
		kRPC.SendMessage(conn, lneup)
		println(stdout, norm(sfrc-ifrc)/norm(ifrc))
		push!(p, 1, aoa, mach)
		gui()
	end
end)
p
while true
	a = readline(stdin)
	if a == "e"
		drags = [if !isnan(dot(cforce[i], drg[i])) dot(cforce[i], drg[i]) else 0.0 end for i=1:length(cforce)]
		lifts = [if !isnan(dot(cforce[i], lft[i])) dot(cforce[i], lft[i]) else 0.0 end for i=1:length(cforce)]
		other = [if !isnan(dot(cforce[i], cross(lft[i],drg[i]))) dot(cforce[i], cross(lft[i],drg[i])) else 0.0 end for i=1:length(cforce)]
		op = DataFrame(aoa=aoal,mach=machl,drag=drags,lift=lifts,other=other)
		CSV.write("lift_drag_test.csv", op)
		break
	end
end