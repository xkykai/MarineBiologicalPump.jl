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

binsize = 16
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
                      empty=[sum(bin_index .== i) > 1 for i in eachindex(bins)]))
end

bin_data = get_age_bin.(Ref(bins), particles_timeseries)

n_start = 700
n_end = Nt
is_top_bottomₙ = vcat([[bin == 1 || bin == nbins for bin in bin_data[n].index] for n in n_start:n_end]...)
selected_bin_categories = vcat([bins[bin_data[n].index] for n in n_start:n_end]...)[.!is_top_bottomₙ]
selected_age_normalized = vcat([bin_data[n].age_normalized for n in n_start:n_end]...)[.!is_top_bottomₙ]
selected_radius_normalized = vcat([bin_data[n].radius for n in n_start:n_end]...)[.!is_top_bottomₙ]

binlocs = bins[end-1:-1:2]
isinmixed_layer = [binloc >= -16 for binloc in binlocs]

fits = [isin ? fit(LogNormal, selected_radius_normalized[selected_bin_categories .≈ binloc]) :
             fit(Pareto, selected_radius_normalized[selected_bin_categories .≈ binloc]) for (isin, binloc) in zip(isinmixed_layer, binlocs)]

fits = [fit(Pareto, selected_radius_normalized[selected_bin_categories .≈ binloc]) for (isin, binloc) in zip(isinmixed_layer, binlocs)]

scatter([pareto_fit.α + 1 for pareto_fit in fits], binlocs)

#%%
locs = 1:length(binlocs)
fig = Figure(size=(500*length(locs)/3, 500*3+100))
axs = [Axis(fig[mod1(i, 3), Int(ceil(i / 3))], title="z = $(binlocs[i]) m", xlabel="log10(particle radius)", ylabel="pdf") for i in 1:length(locs)]
axexponent = Axis(fig[mod1(length(locs)+1, 3), Int(ceil((length(locs)+1) / 3))], title="Exponent", xlabel="Particle size spectrum", ylabel="z (m)")

# for (binloc, fitted_dist, ax, isin) in zip(binlocs[locs], fits[locs], axs, isinmixed_layer)
#     if !isin
#         label_str = @sprintf("Pareto fit, α = %.3f, θ = %.3e", fitted_dist.α, fitted_dist.θ)
#         density!(ax, log10.(rand(fitted_dist, 100000)), label=label_str, npoints=100, color=(:red, 0.8))
#     else
#         label_str = @sprintf("Lognormal fit, μ = %.3f, σ = %.3e", fitted_dist.μ, fitted_dist.σ)
#         density!(ax, log10.(rand(fitted_dist, 100000)), label=label_str, npoints=100, color=(:green, 0.8))
#     end
#     axislegend(ax, position=:rt)
#     density!(ax, log10.(selected_radius_normalized[selected_bin_categories .≈ binloc]), label="Data", npoints=100, color=(:blue, 0.5))
# end

for (binloc, fitted_dist, ax) in zip(binlocs[locs], fits[locs], axs)
    label_str = @sprintf("Pareto fit, α = %.3f, θ = %.3e", fitted_dist.α, fitted_dist.θ)
    density!(ax, log10.(rand(fitted_dist, 100000)), label=label_str, npoints=100, color=(:red, 0.8))
    density!(ax, log10.(selected_radius_normalized[selected_bin_categories .≈ binloc]), label="Data", npoints=100, color=(:blue, 0.5))
    axislegend(ax, position=:rt)
end

scatter!(axexponent, [-pareto_fit.α - 1 for pareto_fit in fits[.!isinmixed_layer]], binlocs[.!isinmixed_layer])
vlines!(axexponent, [size_spectrum], color=:black, linestyle=:dash, linewidth=2, label="Initial particle size spectrum")
ylims!(axexponent, extrema(binlocs))
axislegend(axexponent, position=:rt)

Qᵁ = parameters["momentum_flux"]
Qᴮ = parameters["buoyancy_flux"]
n_particles = parameters["n_particles"]

time_str = "Qᵁ = $(Qᵁ) m² s⁻², Qᴮ = $(Qᴮ) m² s⁻³, initial particle size spectrum ~ r^($(size_spectrum)), time-averaging window: $(round(bbar_data.times[n_start]/24/60^2, digits=2)) - $(round(bbar_data.times[n_end]/24/60^2, digits=2)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

save("./Data/$(filename)_pareto_fits.png", fig, px_per_unit=4)
display(fig)
#%%