using Oceananigans
using Oceananigans.Units
using StructArrays
using JLD2
using FileIO
using Printf
using CairoMakie
using Oceananigans.Grids: halo_size
using Oceananigans.Forcings: AdvectiveForcing
using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ
using Oceananigans.Models.LagrangianParticleTracking: ParticleVelocities, ParticleDiscreteForcing
using Random
using Statistics
using Dates
using CUDA: CuArray

Random.seed!(123)

const Lz = 128meter
const Lx = 128meter
const Ly = 128meter

const Nz = 128
const Nx = 128
const Ny = 128

# const Qᵁ = -1e-4
const Qᵁ = 0
const Qᴮ = 1e-6

const Pr = 1
const ν = 1e-5
const κ = ν / Pr

const f = 1e-4

const dbdz = 2e-4
const b_surface = 0

const pickup = true

# const w_sinking = -Lz / (2 * 24 * 60^2)
const w_sinking = 0

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_NAME = "Lagrangian_QU_$(Qᵁ)_QB_$(Qᴮ)_dbdz_$(dbdz)_Lxz_$(Lx)_$(Lz)_w_$(w_sinking)"
FILE_DIR = "LES/$(FILE_NAME)"
mkpath(FILE_DIR)

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

noise(x, y, z) = rand() * exp(z / 8)

b_initial(x, y, z) = dbdz * z + b_surface
b_initial_noisy(x, y, z) = b_initial(x, y, z) + 1e-6 * noise(x, y, z)

b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᴮ), bottom=GradientBoundaryCondition(dbdz))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))

damping_rate = 1/5minutes

b_target(x, y, z, t) = b_initial(x, y, z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
b_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=b_target)

forcings = (; b = b_sponge, u = uvw_sponge, v = uvw_sponge, w = uvw_sponge)
#%%
struct LagrangianPOC{T}
    x :: T
    y :: T
    z :: T
end

n_particles = 1000

x_particle = CuArray(rand(n_particles) * Lx)
y_particle = CuArray(rand(n_particles) * Ly)
z_particle = CuArray(-0.1 * rand(n_particles) * Lz)

particles = StructArray{LagrangianPOC}((x_particle, y_particle, z_particle))

@inline function sinking_dynamics(x, y, z, w_fluid, particles, p, grid, clock, Δt, model_fields)
    return w_sinking + w_fluid
end

particle_forcing_w  = ParticleDiscreteForcing(sinking_dynamics)
particle_velocities = ParticleVelocities(w=particle_forcing_w)

lagrangian_particles = LagrangianParticles(particles, advective_velocity=particle_velocities)

#%%
model = NonhydrostaticModel(; 
            grid = grid,
            closure = ScalarDiffusivity(ν=ν, κ=κ),
            coriolis = FPlane(f=f),
            buoyancy = BuoyancyTracer(),
            tracers = (:b),
            timestepper = :RungeKutta3,
            advection = WENO(order=9),
            boundary_conditions = (b=b_bcs, u=u_bcs),
            forcing = forcings,
            particles = lagrangian_particles
            )

set!(model, b=b_initial_noisy)

b = model.tracers.b
u, v, w = model.velocities

simulation = Simulation(model, Δt=0.1seconds, stop_time=2days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=10minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, <z> %6.3e next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            mean(lagrangian_particles.properties.z),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/parameters/coriolis_parameter"] = f
    file["metadata/parameters/momentum_flux"] = Qᵁ
    file["metadata/parameters/buoyancy_flux"] = Qᴮ
    file["metadata/parameters/dbdz"] = dbdz
    file["metadata/parameters/b_surface"] = b_surface
    return nothing
end

ubar = Average(u, dims=(1, 2))
vbar = Average(v, dims=(1, 2))
bbar = Average(b, dims=(1, 2))

field_outputs = merge(model.velocities, model.tracers)
timeseries_outputs = (; ubar, vbar, bbar)
particle_outputs = (; model.particles)

#=
simulation.output_writers[:xy_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_xy.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          overwrite_existing = true,
                                                          indices = (:, :, Nz))

simulation.output_writers[:yz_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_yz.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          overwrite_existing = true,
                                                          indices = (1, :, :))

simulation.output_writers[:xz_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_xz.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          overwrite_existing = true,
                                                          indices = (:, 1, :))

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          overwrite_existing = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:particles] = JLD2OutputWriter(model, particle_outputs,
                                                          filename = "$(FILE_DIR)/particles.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          overwrite_existing = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1days), prefix="$(FILE_DIR)/model_checkpoint")

if pickup
    files = readdir(FILE_DIR)
    checkpoint_files = files[occursin.("model_checkpoint_iteration", files)]
    if !isempty(checkpoint_files)
        checkpoint_iters = parse.(Int, [filename[findfirst("iteration", filename)[end]+1:findfirst(".jld2", filename)[1]-1] for filename in checkpoint_files])
        pickup_iter = maximum(checkpoint_iters)
        run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration$(pickup_iter).jld2")
    else
        run!(simulation)
    end
else
    run!(simulation)
end
=#
#%%

b_xy_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xy.jld2", "b", backend=OnDisk())
b_xz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_xz.jld2", "b", backend=OnDisk())
b_yz_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields_yz.jld2", "b", backend=OnDisk())

bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar")

particle_data = jldopen("$(FILE_DIR)/particles.jld2", "r") do file
    iters = keys(file["timeseries/t"])
    particle_timeseries = [file["timeseries/particles/$(iter)"] for iter in iters]
    return particle_timeseries
end

blim = (find_min(b_xy_data, b_yz_data, b_xz_data), find_max(b_xy_data, b_yz_data, b_xz_data))
bbarlim = (minimum(bbar_data), maximum(bbar_data))

Nt = length(bbar_data.times)

xC = bbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = bbar_data.grid.xᶜᵃᵃ[1:Ny]
zC = bbar_data.grid.zᵃᵃᶜ[1:Nz]

#%%
fig = Figure(resolution=(2000, 2000))

axb = Axis3(fig[1:2, 1:2], title="b", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)

axbbar = Axis(fig[3, 1], title="<b>", xlabel="<b>", ylabel="z")
axparticle = Axis(fig[3, 2], title="Particle location", xlabel="x", ylabel="z")

xs_xy = xC
ys_xy = yC
zs_xy = [zC[Nz] for x in xs_xy, y in ys_xy]

ys_yz = yC
xs_yz = range(xC[1], stop=xC[1], length=length(zC))
zs_yz = zeros(length(xs_yz), length(ys_yz))
for j in axes(zs_yz, 2)
  zs_yz[:, j] .= zC
end

xs_xz = xC
ys_xz = range(yC[1], stop=yC[1], length=length(zC))
zs_xz = zeros(length(xs_xz), length(ys_xz))
for i in axes(zs_xz, 1)
  zs_xz[i, :] .= zC
end

colormap = Reverse(:RdBu_10)
b_color_range = blim

n = Observable(1)

parameters = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

bₙ_xy = @lift interior(b_xy_data[$n], :, :, 1)
bₙ_yz = @lift transpose(interior(b_yz_data[$n], 1, :, :))
bₙ_xz = @lift interior(b_xz_data[$n], :, 1, :)

bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)
xs_particleₙ = @lift particle_data[$n].x
zs_particleₙ = @lift particle_data[$n].z

b_xy_surface = surface!(axb, xs_xy, ys_xy, zs_xy, color=bₙ_xy, colormap=colormap, colorrange = b_color_range)
b_yz_surface = surface!(axb, xs_yz, ys_yz, zs_yz, color=bₙ_yz, colormap=colormap, colorrange = b_color_range)
b_xz_surface = surface!(axb, xs_xz, ys_xz, zs_xz, color=bₙ_xz, colormap=colormap, colorrange = b_color_range)

lines!(axbbar, bbarₙ, zC)
scatter!(axparticle, xs_particleₙ, zs_particleₙ)

xlims!(axbbar, bbarlim)
xlims!(axparticle, (0, Lx))

ylims!(axparticle, (-Lz, 0))

trim!(fig.layout)
display(fig)

record(fig, "./Data/$(FILE_NAME).mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"

#%%
#%%
# fig = Figure(resolution=(1800, 1500))

# axubar = Axis(fig[1, 1], title="<u>", xlabel="<u>", ylabel="z")
# axvbar = Axis(fig[1, 2], title="<v>", xlabel="<v>", ylabel="z")
# axbbar = Axis(fig[1, 3], title="<b>", xlabel="<b>", ylabel="z")
# axcbar = Axis(fig[2, 1], title="<c>", xlabel="<c>", ylabel="z")

# ubarlim = (minimum(ubar_data), maximum(ubar_data))
# vbarlim = (minimum(vbar_data), maximum(vbar_data))
# bbarlim = (minimum(bbar_data), maximum(bbar_data))
# cbarlim = (find_min(csbar_data...), find_max(csbar_data...))

# n = Observable(1)

# parameters = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
#     return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
# end 

# time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
# title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

# ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
# vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
# bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)

# csbarₙ = [@lift interior(data[$n], 1, 1, :) for data in csbar_data]

# # cmin = @lift find_min([interior(data[$n], 1, 1, :) for data in csbar_data]..., -1e-5)
# # cmax = @lift find_max([interior(data[$n], 1, 1, :) for data in csbar_data]..., 1e-5)

# # cbarlim = @lift (find_min([interior(data[$n], 1, 1, :) for data in csbar_data]..., -1e-5), find_max([interior(data[$n], 1, 1, :) for data in csbar_data]..., 1e-5))

# lines!(axubar, ubarₙ, zC)
# lines!(axvbar, vbarₙ, zC)
# lines!(axbbar, bbarₙ, zC)

# for (i, data) in enumerate(csbarₙ)
#     lines!(axcbar, data, zC, label="c$(i-1)")
# end

# Legend(fig[2, 2], axcbar, tellwidth=false)

# xlims!(axubar, ubarlim)
# xlims!(axvbar, vbarlim)
# xlims!(axbbar, bbarlim)
# xlims!(axcbar, cbarlim)
# # xlims!(axcbar, (cmin[], cmax[]))
# # xlims!(axcbar, (-0.1, 0.1))

# trim!(fig.layout)

# record(fig, "$(FILE_DIR)/$(FILE_NAME).mp4", 1:Nt, framerate=15) do nn
#     n[] = nn
# end

# @info "Animation completed"

# #%%