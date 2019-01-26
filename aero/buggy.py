import krpc
import time
from numpy import linalg as LA

conn = krpc.connect()
vessel = conn.space_center.active_vessel
orbit = vessel.orbit
body = orbit.body
flight = vessel.flight()
simf = conn.add_stream(flight.simulate_aerodynamic_force_at, body, (0.0,0.0,0.0), (0.0,0.0,0.0))
rlf = conn.add_stream(getattr, flight, 'aerodynamic_force')

for i in range(1,100):
	print(LA.norm(simf())/LA.norm(rlf()))
	time.sleep(0.1)