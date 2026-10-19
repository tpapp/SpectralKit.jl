module SpectralKitEnzymeExt

using Enzyme
import Enzyme.EnzymeRules: forward, needs_shadow, needs_primal, augmented_primal, reverse
import SpectralKit as SK

function forward(config::EnzymeRules.FwdConfigWidth{1}, ::Const{typeof(SK._inf_to_pm1)},
                 ::Type{RT}, z::Union{Const,Duplicated},
                 s::Union{Const,Duplicated}) where {RT<:Union{Const,Duplicated,DuplicatedNoNeed}}
    intermediate = SK._inf_to_pm1_intermediate(z.val, s.val)
    if needs_shadow(config)
        dx_dz, dx_ds = SK._inf_to_pm1_partials(intermediate)
        tangent = zero(intermediate.x)
        z isa Duplicated && (tangent += dx_dz * z.dval)
        s isa Duplicated && (tangent += dx_ds * s.dval)
        needs_primal(config) ? Duplicated(intermediate.x, tangent) : tangent
    else
        needs_primal(config) ? intermediate.value : nothing
    end
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth{1},
                                      ::Const{typeof(SK._inf_to_pm1)}, ::Type{RT},
                                      z::Union{Const,Active},
                                      s::Union{Const,Active}) where {RT<:Union{Const,Active}}
    intermediate = SK._inf_to_pm1_intermediate(z.val, s.val)
    primal = EnzymeRules.needs_primal(config) ? intermediate.x : nothing
    EnzymeRules.AugmentedReturn(primal, nothing, intermediate)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth{1},
                             ::Const{typeof(SK._inf_to_pm1)}, dret, intermediate,
                             z::Union{Const,Active}, s::Union{Const,Active})
    dret isa Active || return (nothing, nothing)
    dx_dz, dx_ds = SK._inf_to_pm1_partials(intermediate)
    dz = z isa Active ? dx_dz * dret.val : nothing
    ds = s isa Active ? dx_ds * dret.val : nothing
    (dz, ds)
end

end
