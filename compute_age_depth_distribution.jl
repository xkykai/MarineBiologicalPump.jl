using Oceananigans
using StatsBase
using JLD2
using CairoMakie
using Statistics
using ArgParse
using LsqFit

filename = "Lagrangian_Pareto_decay_alpha_1.0_rmin_0.005_n_particles_50000_A_-1481.4814814814815_QU_-2.0e-5_QB_0.0_dbdz_0.0001_Lxz_128.0_256.0_Nxz_128_256_AMD"
FILE_DIR = "./LES/$(filename)"

particles_data = jldopen("$(FILE_DIR)/particles.jld2", "r")

size_spectrum = -2

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

#%%
#=
fig = Figure(size=(1920, 1080))
axbbar = Axis(fig[1, 1])
axbox = Axis(fig[1, 2])

n = Observable(1000)

bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)

is_top_bottomₙ = @lift [bin == 1 || bin == nbins for bin in bin_data[$n].index]
selected_bin_categories = @lift bins[bin_data[$n].index[.!$is_top_bottomₙ]]
selected_age_normalized = @lift bin_data[$n].age_normalized[.!$is_top_bottomₙ]

mean_age_normalized = @lift mean.(bin_data[$n].data.age_normalized[2:end-1])

line = lines!(axbbar, bbarₙ, zC)

boxplot!(axbox, selected_bin_categories, selected_age_normalized, orientation=:horizontal, show_outliers=false, width=binsize/2, color=line.attributes.color)
scatter!(axbox, mean_age_normalized, bins[2:end-1], color=:red, markersize=15)

display(fig)

record(fig, "./Data/lgp_test.mp4", 1000:Nt, framerate=15) do nn
    @info nn
    n[] = nn
    is_top_bottomₙ = [bin == 1 || bin == nbins for bin in bin_data[nn].index]
    selected_bin_categoriesₙ = bins[bin_data[nn].index[.!is_top_bottomₙ]]
    selected_age_normalizedₙ = bin_data[nn].age_normalized[.!is_top_bottomₙ]
    mean_age_normalized = mean.(bin_data[nn].data.age_normalized[2:end-1])
    empty!(axbox)
    boxplot!(axbox, selected_bin_categoriesₙ, selected_age_normalizedₙ, orientation=:horizontal, show_outliers=false, width=binsize/2, color=line.attributes.color)
    scatter!(axbox, mean_age_normalized, bins[2:end-1], color=:red, markersize=15)
    xlims!(axbox, (0, 5e6))
end
=#
#%%
fig = Figure(size=(1920, 1080))
axbbar = Axis(fig[1, 1], title="<b>", xlabel="Buoyancy (m s⁻²)", ylabel="z (m)")
axbox = Axis(fig[1, 2], title="Time-averaged particle age", xlabel="Mass-weighted age (s)", ylabel="z (m)")
axviolin = Axis(fig[1, 3], title="Time-averaged particle age", xlabel="Mass-weighted age (s)", ylabel="z (m)")

n_start = 900
n_end = Nt
is_top_bottomₙ = vcat([[bin == 1 || bin == nbins for bin in bin_data[n].index] for n in n_start:n_end]...)
selected_bin_categories = vcat([bins[bin_data[n].index] for n in n_start:n_end]...)[.!is_top_bottomₙ]
selected_age_normalized = vcat([bin_data[n].age_normalized for n in n_start:n_end]...)[.!is_top_bottomₙ]

mean_age_normalized = mean.(vcat.([bin_data[n].data.age_normalized[2:end-1] for n in n_start:n_end]...))
median_age_normalized = median.(vcat.([bin_data[n].data.age_normalized[2:end-1] for n in n_start:n_end]...))
p0 = [-10., 20]

bins_curvefit = bins[2:end-1][1:end-1]
mean_age_curvefit = mean_age_normalized[1:end-1]
median_age_curvefit = median_age_normalized[1:end-1]

fit_mean = curve_fit(linear, bins_curvefit, mean_age_curvefit, p0)
fit_median = curve_fit(linear, bins_curvefit, median_age_curvefit, p0)

# bin_data[end].index

line = lines!(axbbar, interior(bbar_data[n_start], 1, 1, :), zC, label="Start")
lines!(axbbar, interior(bbar_data[n_end], 1, 1, :), zC, label="End")
axislegend(axbbar, position=:rb)

# agelim = (0, quantile(selected_age_normalized[selected_bin_categories .== minimum(selected_bin_categories)], 0.99))
agelim = (0, 0.2)

# density!(axparticle, agesₙ)
box = boxplot!(axbox, selected_bin_categories, selected_age_normalized, orientation=:horizontal, show_outliers=false, width=binsize/2, color=line.attributes.color)
# scatter!(axbox, mean_age_normalized, bins[2:end-1], color=:red, markersize=15, label="Mean")
# lines!(axbox, linear(bins_curvefit, fit_mean.param), bins_curvefit, color=:red, label="Mean regression, gradient = $(round(fit_mean.param[1], digits=3))")
lines!(axbox, linear(bins_curvefit, fit_median.param), bins_curvefit, color=:black, label="Median regression, gradient = $(round(fit_median.param[1], digits=3))")
axislegend(axbox, position=:rt)

# violin!(axviolin, selected_bin_categories, selected_age_normalized, orientation=:horizontal, color=line.attributes.color, width=binsize*4, side=:right, datalimits=agelim)
violin!(axviolin, selected_bin_categories, selected_age_normalized, orientation=:horizontal, color=line.attributes.color, width=binsize*4, side=:right)

# linkxaxes!(axbox, axviolin)

Qᵁ = parameters["momentum_flux"]
Qᴮ = parameters["buoyancy_flux"]
n_particles = parameters["n_particles"]

time_str = "Qᵁ = $(Qᵁ) m² s⁻², Qᴮ = $(Qᴮ) m² s⁻³, particle size spectrum ~ r^($(size_spectrum)), time-averaging window: $(round(bbar_data.times[n_start]/24/60^2, digits=2)) - $(round(bbar_data.times[n_end]/24/60^2, digits=2)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)
save("./Data/$(filename)_regression.png", fig, px_per_unit=8)

display(fig)
#%%