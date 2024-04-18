using Oceananigans
using CairoMakie
using JLD2
using Statistics

FILE_DIR = "./LES/Lagrangian_Pareto_decay_alpha_3.0_rmin_0.005_n_particles_100000_A_-1481.4814814814815_QU_-2.0e-5_QB_2.0e-7_dbdz_0.0001_Lxz_256.0_2048.0_Nxz_128_1024_WENO9nu0"
FILE_NAME = "video"

particles_data = jldopen("$(FILE_DIR)/particles.jld2", "r")

iters = keys(particles_data["timeseries/t"])
times = [particles_data["timeseries/t/$(iter)"] for iter in iters]
particles_timeseries = [particles_data["timeseries/particles/$(iter)"] for iter in iters]

parameters = Dict([(key, particles_data["metadata/parameters/$(key)"]) for key in keys(particles_data["metadata/parameters"])])
grid = Dict([(key, particles_data["grid/$(key)"]) for key in keys(particles_data["grid"])])
close(particles_data)

bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")

bbarlim = (minimum(bbar_data), maximum(bbar_data))

Nt = length(bbar_data.times)

Nx = bbar_data.grid.Nx
Ny = bbar_data.grid.Ny
Nz = bbar_data.grid.Nz

xC = bbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = bbar_data.grid.xᶜᵃᵃ[1:Ny]
zC = bbar_data.grid.zᵃᵃᶜ[1:Nz]

n_particles = length(particles_timeseries[1].x)

Lx = grid["Lx"]
Ly = grid["Ly"]
Lz = grid["Lz"]

binsize = 64
bins = -Lz:binsize:0
nbins = length(bins)

function get_age_bin(bins, obs)
    bin_index = searchsortedlast.(Ref(bins), obs.z)
    z = [obs.z[bin_index .== i] for i in eachindex(bins)]
    age = [obs.age[bin_index .== i] ./ (24 * 60^2) for i in eachindex(bins)]
    radius = [obs.radius[bin_index .== i] for i in eachindex(bins)]
    # mean_age_radius³ = mean((obs.age .* obs.radius .^ 3)[obs.age .!= 0])
    # age_radius³_normalized = [(obs.age[bin_index .== i] .* obs.radius[bin_index .== i].^3) ./ mean_age_radius³ for i in eachindex(bins)]
    mean_radius³ = mean((obs.radius .^ 3)[obs.age .!= 0])
    age_radius³_normalized = [(obs.age[bin_index .== i] .* obs.radius[bin_index .== i].^3) ./ mean_radius³ for i in eachindex(bins)]
    empty = [sum(bin_index .== i) > 1 for i in eachindex(bins)]
    return (; z, age, radius, age_radius³_normalized, empty)
end

binned_ages_data = get_age_bin.(Ref(bins), particles_timeseries)

#%%
fig = Figure(size=(1920, 1080))

axbbar = Axis(fig[1, 1], title="<b>", xlabel="<b>", ylabel="z")
axparticle = Axis(fig[1, 2], title="Particle location", xlabel="x", ylabel="z")
axage = Axis(fig[1, 3], title="Particle age", xlabel="Age (days)", ylabel="z")
axagedist = Axis(fig[1, 4], title="Mass-weighted age distribution, bin size $(binsize) m", xlabel="(Age * radius³) / <Radius³>", ylabel="z", yticks=(1:nbins, string.(bins)))

n = Observable(2)

Qᵁ = parameters["momentum_flux"]
Qᴮ = parameters["buoyancy_flux"]

time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)
xs_particleₙ = @lift particles_timeseries[$n].x
zs_particleₙ = @lift particles_timeseries[$n].z
ages_particleₙ = @lift particles_timeseries[$n].age ./ (24 * 60^2)
# markersizesₙ = @lift 9 * particle_data[$n].radius ./ 0.0015
# markersizesₙ = @lift ifelse($n == 1, 1 .* ones(n_particles), 1 * particles_timeseries[$n].radius .* (particles_timeseries[$n].age .!= 0) ./ mean(particles_timeseries[$n].radius[particles_timeseries[$n].age .!= 0]))
markersizesₙ = @lift ifelse($n == 1, 1 .* ones(n_particles), 50 * particles_timeseries[$n].radius .* (particles_timeseries[$n].age .!= 0) ./ maximum(particles_timeseries[$n].radius[particles_timeseries[$n].age .!= 0]))
# markersizesₙ = @lift ifelse($n == 1, 1 .* ones(n_particles), 1 * particles_timeseries[$n].radius .* (particles_timeseries[$n].age .!= 0))

# depth_categoriesₙ = @lift (1:9)[binned_ages_data[$n].empty][1:end-1]
# age_categoriesₙ = @lift (binned_ages_data[$n].age_radius³)[binned_ages_data[$n].empty][1:end-1]

line = lines!(axbbar, bbarₙ, zC)
scatter!(axparticle, xs_particleₙ, zs_particleₙ, markersize=markersizesₙ)
scatter!(axage, ages_particleₙ, zs_particleₙ, markersize=markersizesₙ)

# raincloud = rainclouds!(axagedist, depth_categoriesₙ, age_categoriesₙ, clouds=hist, plot_boxplots=true, orientation=:horizontal, markersize=3)
ylims!(axagedist, (0.5, nbins))

xlims!(axbbar, bbarlim)
xlims!(axparticle, (0, Lx))
# xlims!(axage, (0, 2days))

ylims!(axparticle, (-Lz, 0))
ylims!(axage, (-Lz, 0))

display(fig)

CairoMakie.record(fig, "$(FILE_DIR)/$(FILE_NAME)_distribution_nonorm.mp4", 2:Nt, framerate=30) do nn
    n[] = nn
    category_index = (binned_ages_data[nn].empty)[2:end-1]
    depth_categories = (1:length(bins))[2:end-1][category_index]
    age_categories = (binned_ages_data[nn].age_radius³_normalized)[2:end-1][category_index]
    empty!(axagedist)
    rainclouds!(axagedist, depth_categories, age_categories, clouds=hist, plot_boxplots=true, orientation=:horizontal, markersize=3, color=line.attributes.color)
    xlims!(axage, (nothing, nothing))
    # xlims!(axagedist, (-10, 5))
    xlims!(axagedist, (nothing, nothing))
    CairoMakie.trim!(fig.layout)
end

@info "Distribution animation completed"
#%%