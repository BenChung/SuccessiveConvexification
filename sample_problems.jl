module SampleProblems
	using LinearAlgebra
	using ..RocketlandDefns
	using ..Aerodynamics
	function normalize_problem(dp::DescentProblem)::DescentProblem
		Ul = maximum(dp.rIi)
		Ut = dp.tf_guess
		Um = dp.mwet
		return DescentProblem(
			g=dp.g/(Ul/Ut^2), mdry=dp.mdry/Um, mwet=dp.mwet/Um,
			Tmin = dp.Tmin/(Um * Ul/Ut^2), Tmax = dp.Tmax/(Um * Ul/Ut^2),
			omMax = dp.omMax/Ut, jB = broadcast(*, dp.jB, 1/(Um * Ul^2)),
			rTB = broadcast(*, dp.rTB, 1/Ul), rIi = broadcast(*, dp.rIi, 1/Ul),
			rIf = broadcast(*, dp.rIf, 1/Ul), vIi = broadcast(*, dp.vIi, 1/(Ul/Ut)),
			vIf = broadcast(*, dp.vIi, 1/(Ul/Ut)), qBIf = dp.qBIf, qBIi = dp.qBIi,
			wBi = dp.wBi, wBf = dp.wBf, rFB = broadcast(*, dp.rFB, 1/Ut),
			deltaMax = dp.deltaMax, thetaMax = dp.thetaMax, gammaGs = dp.gammaGs,
			alpha = dp.alpha/(Ut^2/Ul), K=dp.K, imax = dp.imax, wNu = dp.wNu, wID = dp.wID,
			wDS = dp.wDS, wCst = dp.wCst, wTviol = dp.wTviol, delTol = dp.delTol,
			tf_guess = dp.tf_guess/Ut, ri = dp.ri, rh0 = dp.rh0, rh1 = dp.rh1,
			rh2 = dp.rh2, alph = dp.alph, bet = dp.bet, dpMax=dp.dpMax/(Um/(Ul*Ut^2)), rho = dp.rho/(Um/Ul^3), sos=dp.sos/(Ul/Ut),
			aero = Aerodynamics.rescale_aerodata(dp.aero, Ul, Ut, Um))
	end

	aero_info = Aerodynamics.load_aerodata("aero/lift_drag.csv")
	base_prob = DescentProblem(g=9.82, mwet=66018, mdry=65947, Tmin=0.1*4.686588e6, Tmax=4.686588e6, jB=diagm(0=>[72487.03125,2.0734175e6,2.0734175e6]), 
										  alpha=0.000345, rTB=[-4.26114,0,0], rFB = [2.0,0,0], rIi = [1000.0,1000.0,100.0], rIf=[0.0,0.0,0.0], vIi = [-100.0,-200.0,0], sos=352.0, wNu = 1e4)
	base_prob_scaled = normalize_problem(base_prob)

	base_prob_aero = DescentProblem(g=9.82, mwet=66018, mdry=65947, Tmin=0.1*4.686588e6, Tmax=4.686588e6, jB=diagm(0=>[72487.03125,2.0734175e6,2.0734175e6]), 
										  alpha=0.000345, rTB=[-4.26114,0,0], rFB = [2.0,0,0], rIi = [1000.0,1000.0,100.0], rIf=[0.0,0.0,0.0], vIi = [-100.0,-200.0,0], sos=352.0, aero = aero_info, wNu = 1e4)
	base_prob_aero_scaled = normalize_problem(base_prob_aero)
end