using kRPC
using CSV
using LinearAlgebra
using DataFrames
using Plots
using Rotations
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
ref = ReferenceFrame(vessel)
gfl = Flight(vessel, referenceFrame=rf)


trq_vis = kRPC.Remote.Drawing.AddLine((0.0,0.0,0.0),(0.0,0.0,0.0),rf)
strq_vis = kRPC.Remote.Drawing.AddLine((0.0,0.0,0.0),(0.0,0.0,0.0),rf)
sfrc_vis = kRPC.Remote.Drawing.AddLine((0.0,0.0,0.0),(0.0,0.0,0.0),rf)
kRPC.Remote.Drawing.Color!(trq_vis, (0.0,1.0,0.0))
kRPC.Remote.Drawing.Color!(strq_vis, (0.0,0.0,1.0))
global lut = -10000.0
global lav = Float64[]
global iter = 0
add_multiple_streams([
	kRPC.Remote.SpaceCenter.Delayed.AerodynamicTorque(gfl),
	kRPC.Remote.SpaceCenter.Delayed.Velocity(gfl),
	kRPC.Remote.SpaceCenter.Delayed.Position(vessel, rf)],
	function (at, vel, pos) 
		if iter % 10 == 0
			sat,saf = kRPC.SendMessage(conn, [
				kRPC.Remote.SpaceCenter.Delayed.SimulateAerodynamicTorqueAt(gfl, body, pos, vel),
				kRPC.Remote.SpaceCenter.Delayed.SimulateAerodynamicForceAt(gfl, body, pos, vel)])
			kRPC.SendMessage(conn, 
				[kRPC.Remote.Drawing.Delayed.End!(trq_vis, pos.+at),
			 	 kRPC.Remote.Drawing.Delayed.End!(strq_vis, pos.+sat),
			 	 kRPC.Remote.Drawing.Delayed.End!(sfrc_vis, pos.+saf),
			 	 kRPC.Remote.Drawing.Delayed.Start!(trq_vis, pos),
			 	 kRPC.Remote.Drawing.Delayed.Start!(strq_vis, pos),
			 	 kRPC.Remote.Drawing.Delayed.Start!(sfrc_vis, pos)])
		end
		global iter = iter + 1
	end)

