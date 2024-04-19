using Oceananigans
using StatsBase
using JLD2
using CairoMakie
using Statistics
using ArgParse
using LsqFit

filename = "Lagrangian_Pareto_decay_alpha_3.0_rmin_0.005_n_particles_99999_A_-1481.4814814814815_QU_0.0_QB_2.0e-7_dbdz_0.0001_Lxz_256.0_2048.0_Nxz_128_1024_WENO9nu0"
FILE_DIR = "./LES/$(filename)"
# FILE_DIR = "/storage6/xinkai/MarineBiologicalPump.jl/$(filename)"

particles_data = jldopen("$(FILE_DIR)/particles.jld2", "r")

size_spectrum = -4

iters = keys(particles_data["timeseries/t"])
times = [particles_data["timeseries/t/$(iter)"] for iter in iters]
particles_timeseries = [particles_data["timeseries/particles/$(iter)"] for iter in iters]

parameters = Dict([(key, particles_data["metadata/parameters/$(key)"]) for key in keys(particles_data["metadata/parameters"])])
grid = Dict([(key, particles_data["grid/$(key)"]) for key in keys(particles_data["grid"])])
close(particles_data)

bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")

bbarlim = (minimum(bbar_data), maximum(bbar_data))

Nt = length(bbar_data.times)

Nx = grid["Nx"]
Ny = grid["Ny"]
Nz = grid["Nz"]

Lx = grid["Lx"]
Ly = grid["Ly"]
Lz = grid["Lz"]

xC = bbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = bbar_data.grid.xᶜᵃᵃ[1:Ny]
zC = bbar_data.grid.zᵃᵃᶜ[1:Nz]
zF = bbar_data.grid.zᵃᵃᶠ[1:Nz+1]

binsize = 16
bins = collect(-Lz:binsize:0)
nbins = length(bins)
n_particles = length(particles_timeseries[1].x)

function get_age_bin(bins, obs)
    bin_index = searchsortedlast.(Ref(bins), obs.z)
    mean_radius³ = mean((obs.radius .^ 3)[obs.age .!= 0])
    return (index=bin_index, age_normalized=obs.age .* obs.radius .^ 3 ./ mean_radius³,
            data=(; z=[obs.z[bin_index .== i] for i in eachindex(bins)],
            age=[obs.age[bin_index .== i] ./ (24 * 60^2) for i in eachindex(bins)], 
            radius=[obs.radius[bin_index .== i] for i in eachindex(bins)],
            mean_radius³=mean_radius³,
            age_normalized = [(obs.age[bin_index .== i] .* obs.radius[bin_index .== i].^3) ./ mean_radius³ for i in eachindex(bins)],
            empty=[sum(bin_index .== i) > 1 for i in eachindex(bins)]))
end

bin_data = get_age_bin.(Ref(bins), particles_timeseries)

function linear(x, p)
    return p[1] .* x .+ p[2]
end

tstart = 1500

masses = zeros(Nt - tstart+1, n_particles)
bin_indices = zeros(Nt - tstart + 1, n_particles)

for n in tstart:Nt, i in 1:n_particles
    masses[n - tstart+1, i] = particles_timeseries[n].radius[i]^3 / 5e-3^3
    bin_indices[n - tstart+1, i] = bin_data[n].index[i]
end

is_top_bottom = [bin == 1 || bin == nbins for bin in bin_indices]

masses = vec(masses[.!is_top_bottom])
bin_indices = vec(bin_indices[.!is_top_bottom])

total_mass = zeros(length(2:nbins-1))
for i in 2:nbins-1
    total_mass[i-1] = sum(masses[bin_indices .== i]) + 1e-16
end
#%%
depth_plot = -bins[2:end-1]
mass_plot = total_mass[1:end]

#%%
depth_fit = depth_plot[depth_plot .>= 1000]
mass_fit = mass_plot[depth_plot .>= 1000]

p1 = [-1., 0]
fit_mean = curve_fit(linear, log10.(depth_fit), log10.(mass_fit), p1)

fit_param = fit_mean.param

fit_plot = 10^fit_param[2] .* depth_fit .^ fit_param[1]

MLD= -zF[2:end-1][argmax(diff(interior(bbar_data[Nt], 1, 1, :)))]

#%%
Qᵁ = parameters["momentum_flux"]
Qᴮ = parameters["buoyancy_flux"]
n_particles = parameters["n_particles"]

time_str = "Qᵁ = $(Qᵁ) m² s⁻², Qᴮ = $(Qᴮ) m² s⁻³, particle size spectrum ~ r^($(size_spectrum)) \nTime-averaging window: $(round(bbar_data.times[tstart]/24/60^2, digits=2)) - $(round(bbar_data.times[Nt]/24/60^2, digits=2)) days"

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Depth (m)", ylabel="Total POC mass (a.u.)", yscale=log10, xscale=log10, title=time_str)
scatter!(ax, depth_plot, mass_plot)
lines!(ax, depth_fit, fit_plot, color=:red, label="Power law fit, slope = $(round(fit_param[1], digits=3))")
vlines!(ax, [MLD], color=:black, label="Mixed layer depth", linestyle=:dash)
axislegend(ax, position=:lt)
save("./Data/$(filename)_mass_depth.png", fig, px_per_unit=8)
display(fig)
#%%
# display(fig)
#%%
