using Oceananigans
using StatsBase
using JLD2
using CairoMakie
using Statistics
using ArgParse
using LsqFit
using Distributions
using Printf

filename = "Lagrangian_Pareto_decay_alpha_3.0_rmin_0.005_n_particles_50000_A_-1481.4814814814815_QU_-2.0e-5_QB_0.0_dbdz_0.0001_Lxz_128.0_256.0_Nxz_128_256_AMD"
FILE_DIR = "./LES/$(filename)"

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

A = Nx * Ny
binsize = 4
bins = -Lz:binsize:0
nbins = length(bins)

function get_age_bin(bins, obs)
    bin_index = searchsortedlast.(Ref(bins), obs.z)
    mean_radius³ = mean((obs.radius .^ 3)[obs.age .!= 0])
    return (index = bin_index, 
            age_normalized = obs.age .* obs.radius .^ 3 ./ mean_radius³,
            radius = obs.radius,
            data = (; z=[obs.z[bin_index .== i] for i in eachindex(bins)],
                      age=[obs.age[bin_index .== i] ./ (24 * 60^2) for i in eachindex(bins)], 
                      radius=[obs.radius[bin_index .== i] for i in eachindex(bins)],
                      mean_radius³=mean_radius³,
                      age_normalized = [(obs.age[bin_index .== i] .* obs.radius[bin_index .== i].^3) ./ mean_radius³ for i in eachindex(bins)],
                      w_sinking = [obs.w_sinking[bin_index .== i] for i in eachindex(bins)],
                      empty=[sum(bin_index .== i) > 1 for i in eachindex(bins)]))
end

bin_data = get_age_bin.(Ref(bins), particles_timeseries)

function power_law(x, p)
    return p[1] .* x .+ p[2]
end

function minusone_power_law(x, p)
    return -1 .* x .+ p[1]
end

#%%
n_start = 700
n_end = Nt
bin_index = [bin <= -30 for bin in bins]
mass_flux = [[-sum(bin_data[n].data.radius[i] .^ 3 .* bin_data[n].data.w_sinking[i]) / A / binsize for i in eachindex(bins)][bin_index] for n in n_start:n_end]
averaged_mass_flux = mean(mass_flux, dims=1)[1]
iszero_mass_flux = iszero.(averaged_mass_flux)

p0 = [-1., 0]

flux_fit = curve_fit(linear, log10.(-bins[bin_index][.!iszero_mass_flux]), log10.(averaged_mass_flux[.!iszero_mass_flux]), p0)
flux_fit_minusone = curve_fit(minusone_power_law, log10.(-bins[bin_index][.!iszero_mass_flux]), log10.(averaged_mass_flux[.!iszero_mass_flux]), [p0[2]])

#%%
fig = Figure()
ax = Axis(fig[1, 1], ylabel="z (m)", xlabel="Mass flux (kg m⁻² s⁻¹)", xscale=log10, yscale=log10, yreversed=true)
scatter!(ax, averaged_mass_flux[.!iszero_mass_flux], -bins[bin_index][.!iszero_mass_flux])
lines!(ax, 10 .^ flux_fit.param[2] .* ((-bins[bin_index][.!iszero_mass_flux]) .^ flux_fit.param[1]), -bins[bin_index][.!iszero_mass_flux], label="Power law fit, slope = $(flux_fit.param[1])")
lines!(ax, 10 .^ flux_fit_minusone.param[1] .* ((-bins[bin_index][.!iszero_mass_flux]) .^ -1), -bins[bin_index][.!iszero_mass_flux], label="slope = -1")
axislegend(ax, position=:rb)
display(fig)
save("./Data/$(filename)_mass_flux.png", fig, px_per_unit=4)
#%%