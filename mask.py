import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization

def build_wte_masks(p, d_v, d_e, a):
    # masks for embedding matrix

    split = int(a * d_e)

    mask_1, mask_2 = torch.zeros_like(p), torch.zeros_like(p)

    mask_1[:, :split] = 1
    mask_2[:, split:] = 1
            
    return mask_1, mask_2

def build_ln_masks(p, d_e, a):
    # masks for layer norm matrix

    split = int(a * d_e)

    mask_1, mask_2 = torch.zeros_like(p), torch.zeros_like(p)

    mask_1[:split] = 1
    mask_2[split:] = 1
            
    return mask_1, mask_2

def build_attn_masks(p, d_e, a):
    # masks for Q, K, V matrix

    split = int(a * d_e)

    mask_1, mask_2 = torch.zeros_like(p), torch.zeros_like(p)

    mask_1[:split, :] = 1
    mask_2[split:, :] = 1
            
    return mask_1, mask_2

def build_out_masks(p, d_e, a):
    # masks for attention output matrix

    split = int(a * d_e)

    mask_1, mask_2 = torch.zeros_like(p), torch.zeros_like(p)

    mask_1[:, :split] = 1
    mask_2[:, split:] = 1
            
    return mask_1, mask_2

def build_mlp_in_masks(p, d_e, a):
    # masks for mlp input matrix

    split = int(a * d_e)

    mask_1, mask_2 = torch.zeros_like(p), torch.zeros_like(p)

    mask_1[:split, :] = 1
    mask_2[split:, :] = 1
            
    return mask_1, mask_2

def build_mlp_out_masks(p, d_e, a):
    # masks for mlp output matrix

    split = int(a * d_e)

    mask_1, mask_2 = torch.zeros_like(p), torch.zeros_like(p)

    mask_1[:, :split] = 1
    mask_2[:, split:] = 1
            
    return mask_1, mask_2

# registers backward hooks (but doesn't affect forward pass)
def register_hooks(model, a):
    """
    Build two sets of head masks (for L1 and L2) with fractional split a, and register a hook
    for each parameter that will mask out gradients based on the mutable dict `active`.
    Returns active masks, masks for L1, and masks for L2.
    """
    l = model.config.n_layer
    d_e = model.config.n_embd
    d_v = model.config.vocab_size

    name_to_p = dict(model.named_parameters())

    masks_1 = {}
    masks_2 = {}

    active = {}

    p_wte = "transformer.wte.weight"
    masks_wte = build_wte_masks(name_to_p[p_wte], d_v, d_e, a)
    masks_1[p_wte], masks_2[p_wte], active[p_wte] = masks_wte[0], masks_wte[1], masks_wte[0]
    name_to_p[p_wte].register_hook(lambda grad, n=p_wte: grad * active[n])

    p_ln_f = "transformer.ln_f.weight"
    masks_ln_f = build_ln_masks(name_to_p[p_ln_f], d_e, a)
    masks_1[p_ln_f], masks_2[p_ln_f], active[p_ln_f] = masks_ln_f[0], masks_ln_f[1], masks_ln_f[0]
    name_to_p[p_ln_f].register_hook(lambda grad, n=p_ln_f: grad * active[n])

    for layer in range(l):
        p_ln_w_1 = f"transformer.h.{layer}.ln_1.weight"
        p_ln_b_1 = f"transformer.h.{layer}.ln_1.bias"

        p_attn = f"transformer.h.{layer}.attn.c_attn.weight"
        p_out = f"transformer.h.{layer}.attn.c_proj.weight"

        p_mlp_in = f"transformer.h.{layer}.mlp.c_fc.weight"
        p_mlp_out = f"transformer.h.{layer}.mlp.c_proj.weight"

        p_ln_w_2 = f"transformer.h.{layer}.ln_1.weight"
        p_ln_b_2 = f"transformer.h.{layer}.ln_1.bias"

        masks_ln_w_1 = build_ln_masks(name_to_p[p_ln_w_1], d_e, a)
        masks_ln_b_1 = build_ln_masks(name_to_p[p_ln_b_1], d_e, a)

        masks_p_attn = build_attn_masks(name_to_p[p_attn], d_e, a)
        masks_p_out = build_out_masks(name_to_p[p_out], d_e, a)

        masks_p_mlp_in = build_mlp_in_masks(name_to_p[p_mlp_in], d_e, a)
        masks_p_mlp_out = build_mlp_out_masks(name_to_p[p_mlp_out], d_e, a)

        masks_ln_w_2 = build_ln_masks(name_to_p[p_ln_w_2], d_e, a)
        masks_ln_b_2 = build_ln_masks(name_to_p[p_ln_b_2], d_e, a)

        masks_1[p_ln_w_1], masks_2[p_ln_w_2], active[p_ln_w_1] = masks_ln_w_1[0], masks_ln_w_1[1], masks_ln_w_1[0]
        masks_1[p_ln_b_1], masks_2[p_ln_b_1], active[p_ln_b_1] = masks_ln_b_1[0], masks_ln_b_1[1], masks_ln_b_1[0]
        masks_1[p_attn], masks_2[p_attn], active[p_attn] = masks_p_attn[0], masks_p_attn[1], masks_p_attn[0]
        masks_1[p_out], masks_2[p_out], active[p_out] = masks_p_out[0], masks_p_out[1], masks_p_out[0]
        masks_1[p_mlp_in], masks_2[p_mlp_in], active[p_mlp_in] = masks_p_mlp_in[0], masks_p_mlp_in[1], masks_p_mlp_in[0]
        masks_1[p_mlp_out], masks_2[p_mlp_out], active[p_mlp_out] = masks_p_mlp_out[0], masks_p_mlp_out[1], masks_p_mlp_out[0]
        masks_1[p_ln_w_2], masks_2[p_ln_w_2], active[p_ln_w_2] = masks_ln_w_2[0], masks_ln_w_2[1], masks_ln_w_2[0]
        masks_1[p_ln_b_2], masks_2[p_ln_b_2], active[p_ln_b_2] = masks_ln_b_2[0], masks_ln_b_2[1], masks_ln_b_2[0]

        name_to_p[p_ln_w_1].register_hook(lambda grad, n=p_ln_w_1: grad * active[n])
        name_to_p[p_ln_b_1].register_hook(lambda grad, n=p_ln_b_1: grad * active[n])
        name_to_p[p_attn].register_hook(lambda grad, n=p_attn: grad * active[n])
        name_to_p[p_out].register_hook(lambda grad, n=p_out: grad * active[n])
        name_to_p[p_mlp_in].register_hook(lambda grad, n=p_mlp_in: grad * active[n])
        name_to_p[p_mlp_out].register_hook(lambda grad, n=p_mlp_out: grad * active[n])
        name_to_p[p_ln_w_2].register_hook(lambda grad, n=p_ln_w_2: grad * active[n])
        name_to_p[p_ln_b_2].register_hook(lambda grad, n=p_ln_b_2: grad * active[n])

    return active, masks_1, masks_2



# --- Parametrization module: elementwise (Hadamard) mask ---
class HadamardMask(nn.Module):
    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
    def forward(self, W):
        return W * self.mask

def _name_to_module_and_attr(model, param_name):
    """
    Split a fully-qualified param name into (module, attr).
    E.g. "transformer.h.0.attn.c_attn.weight" -> (module=..., attr="weight")
    """
    *mod_path, attr = param_name.split(".")
    mod_path = ".".join(mod_path)
    # Build a dict once for speed if you like; here we make it ad hoc for clarity.
    name_to_module = dict(model.named_modules())
    return name_to_module[mod_path], attr

def register_parametrizations(model, a):
    """
    Build two sets of masks (L1/L2) with fractional split 'a' and register
    a HadamardMask parametrization on each targeted parameter (weight/bias).
    Returns:
        controller: object with controller.set_active("L1"/"L2") to flip masks
        masks_1, masks_2: dict[param_name] -> mask tensor (same dtype/shape as param)
    """
    l = model.config.n_layer
    d_e = model.config.n_embd
    d_v = model.config.vocab_size

    name_to_p = dict(model.named_parameters())
    masks_1, masks_2 = {}, {}

    p_wte = "transformer.wte.weight"
    m_wte = build_wte_masks(name_to_p[p_wte], d_v, d_e, a)
    masks_1[p_wte], masks_2[p_wte] = m_wte[0], m_wte[1]

    p_ln_f = "transformer.ln_f.weight"
    m_ln_f = build_ln_masks(name_to_p[p_ln_f], d_e, a)
    masks_1[p_ln_f], masks_2[p_ln_f] = m_ln_f[0], m_ln_f[1]

    for layer in range(l):
        # LN1
        p_ln_w_1 = f"transformer.h.{layer}.ln_1.weight"
        p_ln_b_1 = f"transformer.h.{layer}.ln_1.bias"
        m_ln_w_1 = build_ln_masks(name_to_p[p_ln_w_1], d_e, a)
        m_ln_b_1 = build_ln_masks(name_to_p[p_ln_b_1], d_e, a)
        masks_1[p_ln_w_1], masks_2[p_ln_w_1] = m_ln_w_1[0], m_ln_w_1[1]
        masks_1[p_ln_b_1], masks_2[p_ln_b_1] = m_ln_b_1[0], m_ln_b_1[1]

        # Attention projections
        p_attn = f"transformer.h.{layer}.attn.c_attn.weight"
        p_out  = f"transformer.h.{layer}.attn.c_proj.weight"
        m_attn = build_attn_masks(name_to_p[p_attn], d_e, a)
        m_out  = build_out_masks(name_to_p[p_out],   d_e, a)
        masks_1[p_attn], masks_2[p_attn] = m_attn[0], m_attn[1]
        masks_1[p_out],  masks_2[p_out]  = m_out[0],  m_out[1]

        # MLP projections
        p_mlp_in  = f"transformer.h.{layer}.mlp.c_fc.weight"
        p_mlp_out = f"transformer.h.{layer}.mlp.c_proj.weight"
        m_mlp_in  = build_mlp_in_masks(name_to_p[p_mlp_in],   d_e, a)
        m_mlp_out = build_mlp_out_masks(name_to_p[p_mlp_out], d_e, a)
        masks_1[p_mlp_in],  masks_2[p_mlp_in]  = m_mlp_in[0],  m_mlp_in[1]
        masks_1[p_mlp_out], masks_2[p_mlp_out] = m_mlp_out[0], m_mlp_out[1]

        # LN2
        p_ln_w_2 = f"transformer.h.{layer}.ln_2.weight"
        p_ln_b_2 = f"transformer.h.{layer}.ln_2.bias"
        m_ln_w_2 = build_ln_masks(name_to_p[p_ln_w_2], d_e, a)
        m_ln_b_2 = build_ln_masks(name_to_p[p_ln_b_2], d_e, a)
        masks_1[p_ln_w_2], masks_2[p_ln_w_2] = m_ln_w_2[0], m_ln_w_2[1]
        masks_1[p_ln_b_2], masks_2[p_ln_b_2] = m_ln_b_2[0], m_ln_b_2[1]

    # --- Register parametrizations ---
    per_param = {}  # name -> dict with pm (parametrization), M1, M2, MALL

    with torch.no_grad():
        for pname, p in name_to_p.items():
            if pname not in masks_1:
                continue

            M1 = masks_1[pname].to(p.device, dtype=p.dtype)
            M2 = masks_2[pname].to(p.device, dtype=p.dtype)
            MALL = torch.ones_like(M1)  # all active

            mod, attr = _name_to_module_and_attr(model, pname)
            pm = HadamardMask(M1.clone())  # start in L1
            register_parametrization(mod, attr, pm)

            per_param[pname] = {"pm": pm, "M1": M1, "M2": M2, "MALL": MALL, "module": mod, "attr": attr}

    # --- Controller ---
    class _Controller:
        def __init__(self, records):
            self.records = records
            self._active = "L1"

        @torch.no_grad()
        def set_active(self, tag: str):
            # tag in {"L1","L2","ALL"}
            if tag not in ("L1", "L2", "ALL"):
                raise ValueError("tag must be 'L1', 'L2', or 'ALL'")
            use = "M1" if tag == "L1" else ("M2" if tag == "L2" else "MALL")
            for rec in self.records.values():
                rec["pm"].mask.copy_(rec[use])
            self._active = tag

        def get_active(self):
            return self._active

        def masked_params(self):
            return list(self.records.keys())

    controller = _Controller(per_param)
    controller.set_active("L1")

    return controller, masks_1, masks_2