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
	dfun = derivs[key]

	return Expr(:call, dfun, [_convert_diff(Expr, a, derivs, substs) for a in fnargs]...)
end

SymEngine.N(b::SymEngine.BasicType{Val{:Integer}}) = convert(Float64, convert(BigInt, b))
SymEngine.N(b::SymEngine.BasicType{Val{:Rational}}) = N(numerator(b))/N(denominator(b))
function _convert_diff(::Type{Expr}, ex::Basic, derivs::Dict{Deriv, Symbol}, substs::Dict{Symbol, Any})
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
    	bdy = _convert_diff(Expr, as[1], derivs, substs)
    	pow = _convert_diff(Expr, as[2], derivs, substs)
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
    	nsubsts[_convert_diff(Expr, args[2], derivs, substs)] = 
    		_convert_diff(Expr, args[3], derivs, substs)
    	return _convert_diff(Expr, args[1], derivs, nsubsts)
    end

    as = SymEngine.get_args(ex)

    Expr(:call, SymEngine.map_fn(fn, SymEngine.fn_map),
    	 [_convert_diff(Expr,a,derivs,substs) for a in as]...)
end


function convert_diff(::Type{Expr}, ex::Basic, derivs::Dict{Deriv, Symbol}, substs::Dict{Symbol, Any})
    fn = SymEngine.get_symengine_class(ex)

    if fn == :Symbol
        return Expr(:call, :*, Symbol(SymEngine.toString(ex)), 1)
    elseif (fn in SymEngine.number_types) || (fn == :Constant)
        return SymEngine.N(ex)
    end

    return _convert_diff(Expr, ex, derivs, substs)
end

function make_simplified(name, fun, nargs)
	# symbolically execute the body
	args = [symbols("x$i") for i = 1:nargs]
	retv = fun(args)
	exprs = map(x -> convert_diff(Expr, x, Dict{Deriv,Symbol}(), Dict{Symbol,Any}()), retv)
	rexpr = Expr(:vect, exprs...)
	body = macroexpand(CommonSubexpressions,:(@cse $rexpr))

	# compute the extra arguments
	need_args = Set{Symbol}()
	map(x -> push!(need_args, x), vcat(map(x -> map(Symbol, SymEngine.free_symbols(x)), retv)...))
	map(x -> delete!(need_args, Symbol(x)), args)
	ordered_args = sort(collect(need_args))
	println(ordered_args)

	# generate the header
	header = map(x -> :($(Symbol(x[2])) = inp[$(x[1])]), enumerate(args))

	# generate the in-place assignment
	assignment = map(x -> :(J[$(x[1])] = $(x[2])), enumerate(body.args[end].args))

	#output the simplified function
	fbody = (Expr(:block, header...,
				 body.args[1:end-1]...,
				 assignment...))
	return :($(name)(J, inp, $(ordered_args...)) = $(fbody.args...))
end

function make_jacobian(fnname, fun, nargs, idfs, ivects)
	dfs = Dict{Deriv,Symbol}(
		[Deriv(c.args[2].args[1], [c.args[2].args[2:end]...]) => c.args[3] for c in idfs.args])
	vects = Dict{Symbol,Tuple{Symbol, Int}}(
		[c.args[2] => (c.args[3].args[1], c.args[3].args[2]) for c in ivects.args])

	args = [symbols("x$i") for i = 1:nargs]
	symbolic = eval(:($fun($args)))
	jacobian = [convert_diff(Expr, diff(symbolic[v],j), dfs, Dict{Symbol, Any}()) 
					for v in 1:length(symbolic), j in args]
	mat = Expr(:vcat, [Expr(:row, jacobian[i,:]...) for i in 1:length(symbolic)]...)
	reped = postwalk(x -> x isa Expr && x.head == :call && haskey(vects, x.args[1]) ? 
		begin (fn,idx)=vects[x.args[1]]; op = :($(fn)($(x.args[2:end]...))[$(idx)]); op end : x, mat)
	csed = macroexpand(CommonSubexpressions,:(@cse $reped))

	# we know that csed is of form assignment... matrix; get the matrix
	fmat = csed.args[end]
	mat_ass = Expr[]
	for (row,els) in enumerate(fmat.args)
		for (col,el) in enumerate(els.args)
			if el != 0.0
				push!(mat_ass, :(J[$row,$col] = $el))
			end
		end
	end
	body = Expr(:block,
			[:($(Symbol(el[2])) = arg[$(el[1])]) for el in enumerate(args)]...,
			csed.args[1:end-1]...,
			mat_ass...)
	return :($(esc(fnname))(t, arg, J) = $(body.args...))
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