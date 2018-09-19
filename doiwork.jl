using Mosek
using MathOptInterface
using MathOptInterfaceMosek

const MOI=MathOptInterface
const SA=MOI.ScalarAffineTerm
const VA=MOI.VectorAffineTerm
model = MosekOptimizer(MSK_IPAR_LOG=1,MSK_IPAR_INFEAS_REPORT_AUTO=1)
x = MOI.addvariable!(model)
cs = MOI.addconstraint!(model, MOI.SingleVariable(x), MOI.LessThan(1.0))
MOI.set!(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 
			    MOI.ScalarAffineFunction([SA(1.0, x)],0.0))
MOI.set!(model, MOI.ObjectiveSense(), MOI.MaxSense)

MOI.optimize!(model)

MOI.set!(model, MOI.ConstraintSet(), cs, MOI.LessThan(2.0))
MOI.optimize!(model)
status = MOI.get(model, MOI.TerminationStatus())
if status != MOI.Success
	println(status)
	if status != MOI.SlowProgress
		return
	end
end	
result_status = MOI.get(model, MOI.PrimalStatus())
if result_status != MOI.FeasiblePoint
	println(result_status)
    error("Solver ran successfully did not return a feasible point. The problem may be infeasible.")
end
MathOptInterface.get(model, MathOptInterface.VariablePrimal(),x)