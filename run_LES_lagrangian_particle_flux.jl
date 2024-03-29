using Oceananigans
using Oceananigans.Units
using StructArrays
using JLD2
using FileIO
using Printf
using CairoMakie
using Oceananigans.Grids: halo_size
using Oceananigans.Models.LagrangianParticleTracking: ParticleVelocities, ParticleDiscreteForcing
using Random
using Statistics
using Dates
using CUDA: CuArray
using KernelAbstractions
using Oceananigans.Architectures: device, architecture
using ArgParse


function parse_commandline()
    s = ArgParseSettings()
  
    @add_arg_table! s begin
      "--QU"
        help = "surface momentum flux (m²/s²)"
        arg_type = Float64
        default = 0.
      "--QB"
        help = "surface buoyancy flux (m²/s³)"
        arg_type = Float64
        default = 0.
      "--b_surface"
        help = "surface buoyancy (m/s²)"
        arg_type = Float64
        default = 0.
      "--dbdz"
        help = "Initial buoyancy gradient (s⁻²)"
        arg_type = Float64
        default = 5e-3 / 256
      "--f"
        help = "Coriolis parameter (s⁻¹)"
        arg_type = Float64
        default = 1e-4
      "--Nz"
        help = "Number of grid points in z-direction"
        arg_type = Int64
        default = 128
      "--Nx"
        help = "Number of grid points in x-direction"
        arg_type = Int64
        default = 256
      "--Ny"
        help = "Number of grid points in y-direction"
        arg_type = Int64
        default = 256
      "--Lz"
        help = "Domain depth"
        arg_type = Float64
        default = 256.
      "--Lx"
        help = "Domain width in x-direction"
        arg_type = Float64
        default = 512.
      "--Ly"
        help = "Domain width in y-direction"
        arg_type = Float64
        default = 512.
      "--dt"
        help = "Initial timestep to take (seconds)"
        arg_type = Float64
        default = 0.1
      "--max_dt"
        help = "Maximum timestep (seconds)"
        arg_type = Float64
        default = 10. * 60
      "--stop_time"
        help = "Stop time of simulation (days)"
        arg_type = Float64
        default = 2.
      "--time_interval"
        help = "Time interval of output writer (minutes)"
        arg_type = Float64
        default = 10.
      "--field_time_interval"
        help = "Time interval of output writer for fields (minutes)"
        arg_type = Float64
        default = 10.
      "--checkpoint_interval"
        help = "Time interval of checkpoint writer (days)"
        arg_type = Float64
        default = 1.
      "--fps"
        help = "Frames per second of animation"
        arg_type = Float64
        default = 15.
      "--pickup"
        help = "Whether to pickup from latest checkpoint"
        arg_type = Bool
        default = true
      "--advection"
        help = "Advection scheme used"
        arg_type = String
        default = "AMD"
      "--file_location"
        help = "Location to save files"
        arg_type = String
        default = "."
      "--n_particles"
        help = "Number of particles to release at regular intervals"
        arg_type = Int64
        default = 2000
      "--w_sinking"
        help = "Sinking velocity of particles (m s⁻¹)"
        arg_type = Float64
        default = -0.00018518518518518518
    end
    return parse_args(s)
end

args = parse_commandline()


Random.seed!(123)

const Lz = args["Lz"]
const Lx = args["Lx"]
const Ly = args["Ly"]

const Nz = args["Nz"]
const Nx = args["Nx"]
const Ny = args["Ny"]

const Qᵁ = args["QU"]
const Qᴮ = args["QB"]

if args["advection"] == "WENO9nu1e-5"
    advection = WENO(order=9)
    const ν, κ = 1e-5, 1e-5/Pr
    closure = ScalarDiffusivity(ν=ν, κ=κ)
elseif args["advection"] == "WENO9nu0"
    advection = WENO(order=9)
    const ν, κ = 0, 0
    closure = nothing
elseif args["advection"] == "WENO9AMD"
    advection = WENO(order=9)
    const ν, κ = 0, 0
    closure = AnisotropicMinimumDissipation()
elseif args["advection"] == "AMD"
    advection = CenteredSecondOrder()
    const ν, κ = 0, 0
    closure = AnisotropicMinimumDissipation()
end
# const Lz = 1meter
# const Lx = 1meter
# const Ly = 1meter

# const Nz = 4
# const Nx = 4
# const Ny = 4

# const Qᵁ = 0
# const Qᴮ = 0
const f = args["f"]
const n_particles = args["n_particles"]

const dbdz = args["dbdz"]
const b_surface = args["b_surface"]

const pickup = args["pickup"]

const w_sinking = args["w_sinking"]
# const w_sinking = 0

const stop_time = args["stop_time"]days

function find_min(a...)
    return minimum(minimum.([a...]))
end

function find_max(a...)
    return maximum(maximum.([a...]))
end

FILE_NAME = "Lagrangian_n_particles_$(n_particles)_QU_$(Qᵁ)_QB_$(Qᴮ)_dbdz_$(dbdz)_Lxz_$(Lx)_$(Lz)_Nxz_$(Nx)_$(Nz)_w_$(w_sinking)_$(args["advection"])"
FILE_DIR = "LES/$(FILE_NAME)"
mkpath(FILE_DIR)

grid = RectilinearGrid(Oceananigans.GPU(), Float64,
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
struct LagrangianPOC{X, T, A}
    x :: X
    y :: X
    z :: X
    release_time :: T
    age :: A
end

release_time = CuArray(range(0, stop=stop_time, length=n_particles))
x_particle = CuArray(rand(n_particles) * Lx)
y_particle = CuArray(rand(n_particles) * Ly)
z_particle = CuArray(zeros(n_particles))
age = CuArray(zeros(n_particles))

# x_particle = rand(n_particles) * Lx
# y_particle = rand(n_particles) * Ly
# z_particle = -0.1 * rand(n_particles) * Lz
# release_time = collect(range(0, stop=2days, length=n_particles))

particles = StructArray{LagrangianPOC}((x_particle, y_particle, z_particle, release_time, age))

@inline function u_sinking_dynamics(x, y, z, u_fluid, particles, p, grid, clock, Δt, model_fields)
    t = clock.time
    release_time = particles.release_time[p]
    return ifelse(t >= release_time, u_fluid, 0)
end

@inline function v_sinking_dynamics(x, y, z, v_fluid, particles, p, grid, clock, Δt, model_fields)
    t = clock.time
    release_time = particles.release_time[p]
    return ifelse(t >= release_time, v_fluid, 0)
end

@inline function w_sinking_dynamics(x, y, z, w_fluid, particles, p, grid, clock, Δt, model_fields)
    t = clock.time
    release_time = particles.release_time[p]
    return ifelse(t >= release_time, w_sinking + w_fluid, 0)
end

particle_forcing_u  = ParticleDiscreteForcing(u_sinking_dynamics)
particle_forcing_v  = ParticleDiscreteForcing(v_sinking_dynamics)
particle_forcing_w  = ParticleDiscreteForcing(w_sinking_dynamics)
particle_velocities = ParticleVelocities(u=particle_forcing_u, v=particle_forcing_v, w=particle_forcing_w)

@kernel function update_particle_age!(particles, clock, Δt)
    p = @index(Global)
    @inbounds begin
        particles.age[p] = ifelse(clock.time >= particles.release_time[p], particles.age[p] + Δt, particles.age[p])
    end
end

function update_lagrangian_particle_age!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    workgroup = min(length(particles), 256)
    worksize = length(particles)

    update_particle_age_kernel! = update_particle_age!(device(arch), workgroup, worksize)
    update_particle_age_kernel!(particles.properties, model.clock, Δt)

    return nothing
end

# lagrangian_particles = LagrangianParticles(particles, advective_velocity=particle_velocities)
lagrangian_particles = LagrangianParticles(particles, advective_velocity=particle_velocities, dynamics=update_lagrangian_particle_age!)

#%%
model = NonhydrostaticModel(; 
            grid = grid,
            closure = closure,
            coriolis = FPlane(f=f),
            buoyancy = BuoyancyTracer(),
            tracers = (:b),
            timestepper = :RungeKutta3,
            advection = advection,
            boundary_conditions = (b=b_bcs, u=u_bcs),
            forcing = forcings,
            particles = lagrangian_particles
            )

set!(model, b=b_initial_noisy)

b = model.tracers.b
u, v, w = model.velocities

simulation = Simulation(model, Δt=0.1seconds, stop_time=stop_time)

wizard = TimeStepWizard(max_change=1.1, max_Δt=10minutes, cfl=0.6)

mutable struct ParticleTimeStepWizard{W, RT, T}
    wizard :: W
    release_times :: RT
    next_time_index :: T
end

@inline function (particle_wizard::ParticleTimeStepWizard)(simulation)
    wizard_Δt = particle_wizard.wizard(simulation)
    particle_Δt = particle_wizard.release_times[particle_wizard.next_time_index] - simulation.model.clock.time
    simulation_stopped = simulation.model.clock.time >= simulation.stop_time || simulation.model.clock.iteration >= simulation.stop_iteration

    while particle_Δt == 0 && !simulation_stopped
        # @info "while loop triggered"
        particle_wizard.next_time_index = min(length(particle_wizard.release_times), particle_wizard.next_time_index + 1)
        particle_Δt = particle_wizard.release_times[particle_wizard.next_time_index] - simulation.model.clock.time
    end

    if particle_Δt < wizard_Δt
        simulation.Δt = particle_Δt
        particle_wizard.next_time_index = min(length(particle_wizard.release_times), particle_wizard.next_time_index + 1)
    else
        simulation.Δt = wizard_Δt
    end
end

particle_wizard = ParticleTimeStepWizard(wizard, Array(release_time), 1)
simulation.callbacks[:particle_wizard] = Callback(particle_wizard, IterationInterval(1))

# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("%s [%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, (<x>, <y>, <z>) (%6.3e, %6.3e, %6.3e) next Δt: %s\n",
            Dates.now(),
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            mean(lagrangian_particles.properties.x),
            mean(lagrangian_particles.properties.y),
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

simulation.output_writers[:xy_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_xy.jld2",
                                                          schedule = TimeInterval(args["field_time_interval"]minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (:, :, Nz))

simulation.output_writers[:yz_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_yz.jld2",
                                                          schedule = TimeInterval(args["field_time_interval"]minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (1, :, :))

simulation.output_writers[:xz_jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields_xz.jld2",
                                                          schedule = TimeInterval(args["field_time_interval"]minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!,
                                                          indices = (:, 1, :))

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(args["time_interval"]minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:particles] = JLD2OutputWriter(model, particle_outputs,
                                                          filename = "$(FILE_DIR)/particles.jld2",
                                                          schedule = TimeInterval(args["time_interval"]minutes),
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
fig = Figure(size=(2200, 2000))

axb = Axis3(fig[1:3, 1:3], title="b", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom, aspect=:data)

axbbar = Axis(fig[4, 1], title="<b>", xlabel="<b>", ylabel="z")
axparticle = Axis(fig[4, 2], title="Particle location", xlabel="x", ylabel="z")
axage = Axis(fig[4, 3], title="Particle age", xlabel="Age (s)", ylabel="z")

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
ages_particleₙ = @lift particle_data[$n].age

b_xy_surface = surface!(axb, xs_xy, ys_xy, zs_xy, color=bₙ_xy, colormap=colormap, colorrange = b_color_range)
b_yz_surface = surface!(axb, xs_yz, ys_yz, zs_yz, color=bₙ_yz, colormap=colormap, colorrange = b_color_range)
b_xz_surface = surface!(axb, xs_xz, ys_xz, zs_xz, color=bₙ_xz, colormap=colormap, colorrange = b_color_range)

lines!(axbbar, bbarₙ, zC)
scatter!(axparticle, xs_particleₙ, zs_particleₙ)
scatter!(axage, ages_particleₙ, zs_particleₙ)

xlims!(axbbar, bbarlim)
xlims!(axparticle, (0, Lx))

ylims!(axparticle, (-Lz, 0))
ylims!(axage, (-Lz, 0))

trim!(fig.layout)
display(fig)

record(fig, "./Data/$(FILE_NAME).mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

@info "Animation completed"
