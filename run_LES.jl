using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
using CairoMakie
using Oceananigans.Grids: halo_size
using Oceananigans.Forcings: AdvectiveForcing
using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ
using Random
using Statistics
using Dates

Random.seed!(123)

const Lz = 128meter    # depth [m]
const Lx = 64meter
const Ly = 64meter

const Nz = 64
const Nx = 32
const Ny = 32

# const Nz = 32
# const Nx = 64
# const Ny = 64

const Qᵁ = 0
const Qᴮ = 1e-6
const Qᶜ = -2e-6

const Pr = 1
const ν = 1e-5
const κ = ν / Pr

const f = 1e-4

const dbdz = 2e-4
const b_surface = 0

const Nages = 20

const pickup = true
const c0_flux_stop = 2 # day where carbon flux decreases to middle of tanh function

const Δa = 10 * 60 # 10 minutes age
const w_sinking = -Lz / (4 * 24 * 60^2)

FILE_NAME = "QU_$(Qᵁ)_QB_$(Qᴮ)_dbdz_$(dbdz)_Nages_$(Nages)_Lxz_$(Lx)_$(Lz)_halfc0_$(c0_flux_stop)_w_$(w_sinking)_test"
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


@inline c0_flux(x, y, t) = Qᶜ * (-tanh(t - c0_flux_stop) + 1)

c0_top_bc = FluxBoundaryCondition(c0_flux)

b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᴮ), bottom=GradientBoundaryCondition(dbdz))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))
c0_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(c0_flux))

damping_rate = 1/5minutes

b_target(x, y, z, t) = b_initial(x, y, z)

bottom_mask = GaussianMask{:z}(center=-grid.Lz, width=grid.Lz/10)

uvw_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)
b_sponge = Relaxation(rate=damping_rate, mask=bottom_mask, target=b_target)
c_sponge = Relaxation(rate=damping_rate, mask=bottom_mask)

sinking = AdvectiveForcing(w=w_sinking)

# const Nages = 4

tracer_index = 0
forcing_c = Symbol(:forcing_c, tracer_index)
c  = Symbol(:c, tracer_index)
cᴿ¹ = Symbol(:c, tracer_index + 1)
cᴿ² = Symbol(:c, tracer_index + 2)
@eval begin
    @inline function $forcing_c(i, j, k, grid, clock, fields)
        @inbounds begin
            c = fields.$c[i, j, k]
            cᴿ¹ = fields.$cᴿ¹[i, j, k]
            cᴿ² = fields.$cᴿ²[i, j, k]

            return -(-3c + 4cᴿ¹ - cᴿ²) / (2 * Δa) - c / (clock.time + 1e-8)
        end
    end
    c_forcings = (; $c = (Forcing($forcing_c, discrete_form=true), sinking, c_sponge))
end

for tracer_index in 1:Nages - 2
    forcing_c = Symbol(:forcing_c, tracer_index)
    c  = Symbol(:c, tracer_index)
    cᴸ = Symbol(:c, tracer_index - 1)
    cᴿ = Symbol(:c, tracer_index + 1)
    @eval begin
        @inline function $forcing_c(i, j, k, grid, clock, fields)
            @inbounds begin
                c = fields.$c[i, j, k]
                cᴸ = fields.$cᴸ[i, j, k]
                cᴿ = fields.$cᴿ[i, j, k]

                return -(cᴿ - cᴸ) / (2 * Δa) - c / (clock.time + 1e-8)
            end
        end
        c_forcings = merge(c_forcings, (; $c = (Forcing($forcing_c, discrete_form=true), sinking, c_sponge)))
    end
end

tracer_index = Nages - 1
forcing_c = Symbol(:forcing_c, tracer_index)
c  = Symbol(:c, tracer_index)
cᴸ¹ = Symbol(:c, tracer_index - 1)
cᴸ² = Symbol(:c, tracer_index - 2)
@eval begin
    @inline function $forcing_c(i, j, k, grid, clock, fields)
        @inbounds begin
            c = fields.$c[i, j, k]
            cᴸ¹ = fields.$cᴸ¹[i, j, k]
            cᴸ² = fields.$cᴸ²[i, j, k]

            return -(3c - 4cᴸ¹ + cᴸ²) / (2 * Δa) - c / (clock.time + 1e-8)
        end
    end
    c_forcings = merge(c_forcings, (; $c = (Forcing($forcing_c, discrete_form=true), sinking, c_sponge)))
end

tracers = [Symbol(:c, i) for i in 0:Nages - 1]
tracers = push!(tracers, :b)
tracers = Tuple(tracers)

forcings = merge(c_forcings, (; b = b_sponge, u = uvw_sponge, v = uvw_sponge, w = uvw_sponge))

model = NonhydrostaticModel(; 
            grid = grid,
            closure = ScalarDiffusivity(ν=ν, κ=κ),
            coriolis = FPlane(f=f),
            buoyancy = BuoyancyTracer(),
            tracers = tracers,
            timestepper = :RungeKutta3,
            advection = WENO(order=9),
            boundary_conditions = (b=b_bcs, u=u_bcs, c0=c0_bcs),
            forcing = forcings
            )

set!(model, b=b_initial_noisy)

b = model.tracers.b
cs = [model.tracers[Symbol(:c, i)] for i in 0:Nages - 1]
u, v, w = model.velocities

simulation = Simulation(model, Δt=0.1seconds, stop_time=8days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=10minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, max(c0) %6.3e, next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            maximum(abs, sim.model.tracers.c0),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1000))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/parameters/coriolis_parameter"] = f
    file["metadata/parameters/momentum_flux"] = Qᵁ
    file["metadata/parameters/buoyancy_flux"] = Qᴮ
    file["metadata/parameters/carbon_flux"] = Qᶜ
    file["metadata/parameters/dbdz"] = dbdz
    file["metadata/parameters/b_surface"] = b_surface
    file["metadata/parameters/half_c0_flux_time"] = c0_flux_stop
    return nothing
end

ubar = Average(u, dims=(1, 2))
vbar = Average(v, dims=(1, 2))
bbar = Average(b, dims=(1, 2))
cbar_symbols = [Symbol(:c, i, :bar) for i in 0:Nages - 1]
csbar = [Average(c, dims=(1, 2)) for c in cs]

field_outputs = merge(model.velocities, model.tracers)
timeseries_outputs = (; ubar, vbar, bbar)
timeseries_outputs = merge(timeseries_outputs, (; zip(cbar_symbols, csbar)...))

simulation.output_writers[:xy_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_xy.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (:, :, Nz))

simulation.output_writers[:yz_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_yz.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (1, :, :))

simulation.output_writers[:xz_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_xz.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (:, 1, :))

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(10minutes),
                                                          with_halos = true,
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

#%%
ubar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "ubar", backend=OnDisk())
vbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "vbar", backend=OnDisk())
bbar_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "bbar", backend=OnDisk())

csbar_data = [FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "c$(i)bar", backend=OnDisk()) for i in 0:Nages-1]

Nt = length(bbar_data.times)

xC = bbar_data.grid.xᶜᵃᵃ[1:Nx]
yC = bbar_data.grid.xᶜᵃᵃ[1:Ny]
zC = bbar_data.grid.zᵃᵃᶜ[1:Nz]

##
fig = Figure(resolution=(1800, 1500))

axubar = Axis(fig[1, 1], title="<u>", xlabel="<u>", ylabel="z")
axvbar = Axis(fig[1, 2], title="<v>", xlabel="<v>", ylabel="z")
axbbar = Axis(fig[1, 3], title="<b>", xlabel="<b>", ylabel="z")
axcbar = Axis(fig[2, 1], title="<c>", xlabel="<c>", ylabel="z")

function find_min(a...)
  return minimum(minimum.([a...]))
end

function find_max(a...)
  return maximum(maximum.([a...]))
end

ubarlim = (minimum(ubar_data), maximum(ubar_data))
vbarlim = (minimum(vbar_data), maximum(vbar_data))
bbarlim = (minimum(bbar_data), maximum(bbar_data))
cbarlim = (find_min(csbar_data...), find_max(csbar_data...))

n = Observable(1)

parameters = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
    return Dict([(key, file["metadata/parameters/$(key)"]) for key in keys(file["metadata/parameters"])])
end 

time_str = @lift "Qᵁ = $(Qᵁ), Qᴮ = $(Qᴮ), Time = $(round(bbar_data.times[$n]/24/60^2, digits=3)) days"
title = Label(fig[0, :], time_str, font=:bold, tellwidth=false)

ubarₙ = @lift interior(ubar_data[$n], 1, 1, :)
vbarₙ = @lift interior(vbar_data[$n], 1, 1, :)
bbarₙ = @lift interior(bbar_data[$n], 1, 1, :)

csbarₙ = [@lift interior(data[$n], 1, 1, :) for data in csbar_data]

lines!(axubar, ubarₙ, zC)
lines!(axvbar, vbarₙ, zC)
lines!(axbbar, bbarₙ, zC)

for (i, data) in enumerate(csbarₙ)
    lines!(axcbar, data, zC, label="c$(i-1)")
end

Legend(fig[2, 2], axcbar, tellwidth=false)

xlims!(axubar, ubarlim)
xlims!(axvbar, vbarlim)
xlims!(axbbar, bbarlim)
xlims!(axcbar, cbarlim)

trim!(fig.layout)

record(fig, "$(FILE_DIR)/$(FILE_NAME).mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"

#%%