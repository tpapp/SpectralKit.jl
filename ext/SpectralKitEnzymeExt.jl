module SpectralKitEnzymeExt

using Enzyme
import Enzyme.EnzymeRules: forward, needs_shadow, needs_primal, augmented_primal, reverse
import SpectralKit as SK

_d(y::Const, d, i) = false       # 0
_d(y::Duplicated, d, i) = d * y.dval
_d(y::BatchDuplicated, d, i) = d * y.dval[i]

function forward(config::EnzymeRules.FwdConfigWidth{W}, ::Const{typeof(SK._inf_to_pm1)},
                 ::Type{RT}, z::Union{Const,Duplicated,BatchDuplicated},
                 s::Union{Const,Duplicated,BatchDuplicated}) where {W,RT<:Union{Const,Duplicated,DuplicatedNoNeed}}
    intermediate = SK._inf_to_pm1_intermediate(z.val, s.val)
    if needs_shadow(config)
        dx_dz, dx_ds = SK._inf_to_pm1_partials(intermediate)
        shadow = ntuple(i -> _d(z, dx_dz, i) + _d(s, dx_ds, i), Val(W))
        if needs_primal(config)
            if W == 1
                Duplicated(intermediate.x, shadow[1])
            else
                BatchDuplicated(intermediate.x, shadow)
            end
        else
            W == 1 ? shadow[1] : shadow
        end
    else
        needs_primal(config) ? intermediate.x : nothing
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
