using OrdinaryDiffEq
using CairoMakie
using Distributions
using Random
Random.seed!(123)

dist = Pareto(3, 1)
rs = rand(dist, 10000)

function age_limited_decay_mass!(du, u, p, t)
    m, a, z = u
    du[1] = -m / a
    du[2] = 1
    du[3] = p * m^(2/3)
end

u0 = [1, 0.00000001, 0]

tspan = (0.0, 1.0)
times = tspan[1]:0.01:tspan[2]
p = 1
prob = ODEProblem(age_limited_decay_mass!, u0, tspan, p)

sol = solve(prob, Rodas5P())

#%%
fig = Figure()
ax = Axis(fig[1, 1])
# lines!(ax, sol.t, sol[1, :], label="m")
# lines!(ax, sol.t, sol[2, :], label="a")
# lines!(ax, sol.t, sol[3, :], label="z")
lines!(ax, sol[2, :], sol[3, :], xlabel="a", ylabel="z")
# axislegend(ax, position=:rt)
display(fig)
#%%
u0s = ([r^3, 1e-6, 0] for r in rs)
probs = [ODEProblem(age_limited_decay_mass!, u0, tspan, p, saveat=times) for u0 in u0s]

sols = solve.(probs, Ref(Rodas5P()))

mean_mass = mean(sol[1, :] for sol in sols)
age_mass_normalized = [sol[2, :] .* sol[1, :] ./ mean_mass for sol in sols]
mean_age_mass_normalized = [mean([age_mass_normalized[i][t] for i in eachindex(rs)]) for t in eachindex(times)]

#%%
fig = Figure()
ax = Axis(fig[1, 1])
for sol in sols
    lines!(ax, sol[2, :], sol[3, :])
end

display(fig)
#%%
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Normalized age", ylabel="z")
lines!(ax, mean_age_mass_normalized, sols[1][3, :])
display(fig)
#%%

