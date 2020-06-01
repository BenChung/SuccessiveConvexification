module SymbolicUtils
using SymEngine
using CommonSubexpressions
using MacroTools: postwalk
using ..RocketlandDefns

struct Deriv 
	fn::Symbol
	derv::Vector{Int}
end
Base.isequal(a::Deriv, b::Deriv) = (a.fn == b.fn) && (a.derv == b.derv)
Base.:(==)(a::Deriv, b::Deriv) = isequal(a,b)
Base.hash(a::Deriv, h::UInt=0) = hash(a.derv, hash(a.fn, h))
Base.hash(a::Deriv, h::Int=0) = hash(a, convert(UInt, h))

function diff_fun(f::Function, nargs::Int, funs_and_derivs::Dict{Deriv, Symbol})
	args = [symbols("x$i") for i=1:nargs]
	symbolic = f(args...)
	return symbolic
end

fs_rgx = r"(.*?)\("
function get_fnsym_name(ex::Basic)
	return match(fs_rgx, SymEngine.toString(ex))[1]
end

function _convert_deriv(ex::Basic, derivs::Dict{Deriv, Symbol}, substs::Dict{Symbol, Any})
	args = SymEngine.get_args(ex)
	call = args[1]
	fnd = Symbol(get_fnsym_name(call))
	fnargs = SymEngine.get_args(call)
	argo = Dict{Symbol, Int64}(map(p->(Symbol(p[2]),p[1]), enumerate(fnargs)))
	sig = [argo[Symbol(var)] for var in args[2:end]]
	key = Deriv(fnd, sig)
	#dfun = derivs[key]

	return Expr(:ref, Expr(:call, fnd, Val{:jac}, [_convert_diff(a, derivs, substs) for a in fnargs]...), sig[1])
end

SymEngine.N(b::SymEngine.BasicType{Val{:Integer}}) = convert(Float64, convert(BigInt, b))
SymEngine.N(b::SymEngine.BasicType{Val{:Rational}}) = N(numerator(b))/N(denominator(b))
function _convert_diff(ex::Basic, derivs::Dict{Deriv, Symbol}, substs::Dict{Symbol, Any})
    fn = SymEngine.get_symengine_class(ex)
    if fn == :Symbol
    	sym = Symbol(SymEngine.toString(ex))
    	if haskey(substs, sym)
    		return substs[sym]
    	else
    		return sym 
    	end
    elseif (fn in SymEngine.number_types) || (fn == :Constant)
        return N(ex)
    elseif fn == :Pow
    	as = SymEngine.get_args(ex)
    	bdy = _convert_diff(as[1], derivs, substs)
    	pow = _convert_diff(as[2], derivs, substs)
    	if pow == 0.5
    		return :(sqrt($bdy))
    	elseif pow == -0.5
    		return :(1.0/sqrt($bdy))
    	elseif pow == 1.0
    		return bdy
    	elseif pow == 2.0
    		return :($bdy ^ 2)
    	else
    		return :($bdy ^ $pow)
    	end
    elseif fn == :FunctionSymbol
    	fn = Symbol(get_fnsym_name(ex))
    elseif fn == :Derivative
    	return _convert_deriv(ex, derivs, substs)
    elseif fn == :Subs
    	nsubsts = copy(substs)
    	args = SymEngine.get_args(ex)
    	nsubsts[_convert_diff(args[2], derivs, substs)] = 
    		_convert_diff(args[3], derivs, substs)
    	return _convert_diff(args[1], derivs, nsubsts)
    end

    as = SymEngine.get_args(ex)

    Expr(:call, SymEngine.map_fn(fn, SymEngine.fn_map),
    	 [_convert_diff(a,derivs,substs) for a in as]...)
end


function convert_diff(ex::Basic, derivs::Dict{Deriv, Symbol}, substs::Dict{Symbol, Any})
    fn = SymEngine.get_symengine_class(ex)

    if fn == :Symbol
        return Expr(:call, :*, Symbol(SymEngine.toString(ex)), 1)
    elseif (fn in SymEngine.number_types) || (fn == :Constant)
        return SymEngine.N(ex)
    end

    return _convert_diff(ex, derivs, substs)
end

function convert_diff(exes::Array{Basic, D} where D, derivs::Dict{Deriv, Symbol}, substs::Dict{Symbol, Any})
	return map(expr -> convert_diff(expr, derivs, substs), exes)
end

function make_simplified(name, fun, nargs; postprocess = nothing, expected_args = Set{Symbol}())
	# symbolically execute the body
	args = [symbols("x$i") for i = 1:nargs]
	retv = fun(args)
	exprs = convert_diff(retv, Dict{Deriv,Symbol}(), Dict{Symbol,Any}())
	if exprs isa Array{T, 1} where T
		rexpr = Expr(:vect, exprs...)
	elseif exprs isa Array{T, 2} where T
		rexpr = Expr(:vect, reshape(exprs, length(exprs))...)
	else 
		rexpr = Expr(:vect, exprs)
	end
	if !isnothing(postprocess)
		rexpr = postprocess(rexpr)
	end
	body = macroexpand(CommonSubexpressions,:(@cse $rexpr))

	# compute the extra arguments
	need_args = Set{Symbol}()
	map(x -> push!(need_args, x), vcat(map(x -> map(Symbol, SymEngine.free_symbols(x)), retv)...))
	needed_state = intersect(need_args, Set{Symbol}(map(Symbol, args)))
	map(x -> delete!(need_args, Symbol(x)), args)
	ordered_args = sort(collect(union(need_args, expected_args)))

	# generate the header
	header = map((idx,var) -> :($(Symbol(var)) = inp[$idx]), 
		parse.(Int, map(x -> String(x)[2:end], collect(needed_state))), needed_state)

	# generate the in-place assignment
	if exprs isa Array{T, 1} where T
		assignment = map(x -> :(J[$(x[1])] = $(x[2])), enumerate(body.args[end].args))
	elseif exprs isa Array{T, 2} where T
		bargs = reshape(body.args[end].args, size(exprs))
		assignment = vcat([[:(J[$rown, $coln] = $(bargs[rown, coln]))
				for (rown, cell) in enumerate(col) if bargs[rown, coln] != 0.0] 
				for (coln, col) in enumerate(eachcol(exprs))]...)
	else 
		assignment = Expr[:(return $(body.args[end].args...))]
	end
	

	#output the simplified function
	fbody = (Expr(:block, header...,
				 body.args[1:end-1]...,
				 assignment...))

	if exprs isa Array
		return :($(name)(J, inp, $(ordered_args...)) = $(fbody.args...))
	else 
		return :($(name)(inp, $(ordered_args...)) = $(fbody.args...))
	end
end

function make_jacobian(fnname, fun, nargs)
	#=
	dfs = Dict{Deriv,Symbol}(
		[Deriv(c.args[2].args[1], [c.args[2].args[2:end]...]) => c.args[3] for c in idfs.args])
	vects = Dict{Symbol,Tuple{Symbol, Int}}(
		[c.args[2] => (c.args[3].args[1], c.args[3].args[2]) for c in ivects.args])
	;
		postprocess = mat -> postwalk(x -> x isa Expr && x.head == :call && haskey(vects, x.args[1]) ? 
				begin (fn,idx)=vects[x.args[1]]; op = :($(fn)($(x.args[2:end]...))[$(idx)]); op end : x, mat)
		=#
	return make_simplified(fnname, x -> begin frs = fun(x); 
		res = [diff(row, var) for row in frs, var in x];
		return res end, nargs)
end
#=

dp = DescentProblem()
pri = ProbInfo(dp)
@make_simplified(dxs, st -> dx(st[1:14], st[15:17], pri), 17)

@make_jacobian(dx, st -> dx(st[1:14], st[15:17], pri), 17, 
	[ilift[1]=>ilift1, ilift[2]=>ilift2,
	 idrag[1]=>idrag1, idrag[2]=>idrag2],
	[ilift => ilift[1], ilift1 => ilift[2], ilift2 => ilift[3],
	 idrag => idrag[1], idrag1 => idrag[2], idrag2 => idrag[3]])
	 =#
end