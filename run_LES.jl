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

Random.seed!(123)

const Lz = 64meter    # depth [m]
const Lx = 64meter
const Ly = 64meter

const Nz = 32
const Nx = 32
const Ny = 32

# const Nz = 32
# const Nx = 64
# const Ny = 64

const Qᵁ = 0
const Qᴮ = 1e-6
const Qᶜ = 2e-6

const Pr = 1
const ν = 1e-5
const κ = ν / Pr

const f = 1e-4

const λᴮ = 0.4 / Lz

const b_surface = 0

FILE_NAME = "QU_$(Qᵁ)_QB_$(Qᴮ)_test"
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

b_initial(x, y, z) = b_surface * (1 + λᴮ*z) + 1e-6 * noise(x, y, z)

dbdz_bot = λᴮ

b_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᴮ), bottom=GradientBoundaryCondition(dbdz_bot))
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᵁ))
c0_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(Qᶜ))

const Δa = 10 * 60 # 10 minutes age
const w_sinking = 1 / (24 * 60^2)

sinking = AdvectiveForcing(w=w_sinking)

# const Nages = 10
const Nages = 4

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
    c_forcings = (; $c = (Forcing($forcing_c, discrete_form=true), sinking))
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
        c_forcings = merge(c_forcings, (; $c = (Forcing($forcing_c, discrete_form=true), sinking)))
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
    # c_forcings = merge(c_forcings, (; $c = MultipleForcings([Forcing($forcing_c, discrete_form=true), sinking])))
    c_forcings = merge(c_forcings, (; $c = (Forcing($forcing_c, discrete_form=true), sinking)))
end

tracers = [Symbol(:c, i) for i in 0:Nages - 1]
tracers = push!(tracers, :b)
tracers = Tuple(tracers)

model = NonhydrostaticModel(; 
            grid = grid,
            closure = ScalarDiffusivity(ν=ν, κ=κ),
            coriolis = FPlane(f=f),
            buoyancy = BuoyancyTracer(),
            tracers = tracers,
            timestepper = :RungeKutta3,
            advection = WENO(order=9),
            boundary_conditions = (b=b_bcs, u=u_bcs, c0=c0_bcs),
            forcing = c_forcings
            )

set!(model, b=b_initial)
# set!(model, T=20, S=32)

b = model.tracers.b
cs = [model.tracers[Symbol(:c, i)] for i in 0:Nages - 1]
u, v, w = model.velocities

simulation = Simulation(model, Δt=0.1second, stop_time=2days)

wizard = TimeStepWizard(max_change=1.05, max_Δt=10minutes, cfl=0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
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
    file["metadata/parameters/buoyancy_gradient"] = λᴮ
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

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1day), prefix="$(FILE_DIR)/model_checkpoint")

# run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration97574.jld2")
run!(simulation)