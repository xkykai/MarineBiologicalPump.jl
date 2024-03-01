using Oceananigans
using Oceananigans.Fields: get_neutral_mask, condition_operand
using Oceananigans.Fields: ReducedAbstractField, AbstractField
using Oceananigans.Fields: filltype, reduced_location, initialize_reduced_field!
using Oceananigans.Fields: identity, location, indices
using CUDA

import Base

# Allocating and in-place reductions
for reduction in (:sum, :maximum, :minimum, :all, :any, :prod)

    reduction! = Symbol(reduction, '!')

    @eval begin
        
        # In-place
        function Base.$(reduction!)(f::Function,
                                    r::ReducedAbstractField,
                                    a::AbstractField;
                                    condition = nothing,
                                    mask = get_neutral_mask(Base.$(reduction!)),
                                    kwargs...)

            return Base.$(reduction!)(f,
                                      interior(r),
                                      condition_operand(identity, a, condition, mask);
                                      kwargs...)
        end

        function Base.$(reduction!)(r::ReducedAbstractField,
                                    a::AbstractField;
                                    condition = nothing,
                                    mask = get_neutral_mask(Base.$(reduction!)),
                                    kwargs...)

            return Base.$(reduction!)(f,
                                      interior(r),
                                      condition_operand(identity, condition, mask);
                                      kwargs...)
        end

        # Allocating
        function Base.$(reduction)(f::Function,
                                   c::AbstractField;
                                   condition = nothing,
                                   mask = get_neutral_mask(Base.$(reduction!)),
                                   dims = :)

            T = filltype(Base.$(reduction!), c)
            loc = reduced_location(location(c); dims)
            r = Field(loc, c.grid, T; indices=indices(c))
            conditioned_c = condition_operand(identity, c, condition, mask)
            initialize_reduced_field!(Base.$(reduction!), f, r, conditioned_c)
            Base.$(reduction!)(identity, r, conditioned_c, init=false)

            if dims isa Colon
                return CUDA.@allowscalar first(r)
            else
                return r
            end
        end

        Base.$(reduction)(c::AbstractField; kwargs...) = Base.$(reduction)(identity, c; kwargs...)
    end
end