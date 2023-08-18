module ReRoll
import SymbolicRegression: SRRegressor, MultitargetSRRegressor
import MLJ: machine, fit!, predict, report

using Tables

function generate_indices()
    res1 = Float64[]
    res2 = similar(res1)
    res3 = similar(res1)

    for i in 1:10000
        push!(res1, i)
        push!(res2, i - 1)
        push!(res3, i + 1)
    end
    return (x1=res1, x2=res2, x3=res3)
end
export generate_indices

function do_example()
    # Dataset with two named features:
    X = (a=rand(500), b=rand(500))

    # and one target:
    y = @. 2 * cos(X.a * 23.5) - X.b^2

    # with some noise:
    y = y .+ randn(500) .* 1e-3

    model = SRRegressor(
        niterations=50,
        binary_operators=[+, -, *],
        unary_operators=[cos],
    )

    println(Tables.istable(X))
    mach = machine(model, X, y)

    fit!(mach)
    report(mach)
end
export do_example

function do_regression()

    # Dataset with two named features:
    indices = generate_indices()

    # and one target:
    y = (x1=indices[2], x2=indices[3])

    model = MultitargetSRRegressor(
        niterations=50,
        binary_operators=[+, -, *],
    )

    mach = machine(model, ((ind=indices[1],)), y)

    fit!(mach)
    report(mach)
end
export do_regression

end # module ReRoll
