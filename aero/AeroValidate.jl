using Interpolations
using kRPC
using CSV
using LinearAlgebra
using DataFrames
include("../aerodynamics.jl")
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

aerodat = Aerodynamics.load_aerodata("lift_drag.csv",1.0,1.0,1.0)

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
gfl = Flight(vessel, referenceFrame=rf)
sos = Float64(SpeedOfSound(gfl))

add_multiple_streams([kRPC.Remote.SpaceCenter.Delayed.AerodynamicForce(gfl),
	kRPC.Remote.SpaceCenter.Delayed.Direction(gfl),
	kRPC.Remote.SpaceCenter.Delayed.Velocity(gfl),
	kRPC.Remote.SpaceCenter.Delayed.Position(vessel, rf)], function (force,direction,velocity,posn)
	dir,vel = convert(Vector{Float64}, [direction...]), convert(Vector{Float64}, [velocity...])
	aerf,dragf,liftf = Aerodynamics.aero_force(aerodat,dir,vel,sos)
	aoa = acosd(dot(dir,vel)/norm(vel))
	mach = norm(vel)/sos
	liftd = cross(cross(dir, vel),vel)
	liftd = liftd/norm(liftd)
    println("lift ratio: $(liftf/dot(liftd,[force...])) ratio:$(norm(aerf)/norm(force)) dp:$(dot(aerf,force)/(norm(aerf) * norm(force)))")
end)
