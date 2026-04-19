"""Generate SLIC architecture diagram (Condition 1 - Full) for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# ── Color palette ─────────────────────────────────────────────
C_INPUT   = '#E8F4FD'
C_ENC     = '#BDE0FE'
C_SELF    = '#A2D2FF'
C_CROSS   = '#FFC8DD'
C_MLP     = '#FFAFCC'
C_ROLE    = '#FFD166'
C_ARROW   = '#444444'
C_ROLE_ARROW = '#D62828'

def box(ax, x, y, w, h, label, color, fontsize=8, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.04",
                          linewidth=1.2, edgecolor='#333333',
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, zorder=4,
            wrap=True)

def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.2, style='->', label=None):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.05, my, label, fontsize=6.5, color=color, zorder=6)

# ─── Row y-coordinates ────────────────────────────────────────
Y_OBS   = 7.8   # observation tokens (top)
Y_ENC   = 6.3   # encoders
Y_SA    = 4.7   # intra-set self-attention
Y_CA    = 3.1   # cross-attention
Y_MLP   = 1.4   # output MLP
BH      = 0.85  # box height

# ─── Column x-centers ─────────────────────────────────────────
W_WIDE  = 1.9   # wide box width
W_NORM  = 1.7
GAP     = 0.3

xs_center = [1.3, 3.5, 5.7, 9.2]   # self, ally, enemy, zone
xs_label  = ['self_tok\n[8-dim]', 'ally_tok\n[2×8 dim]',
             'enemy_tok\n[2×8 dim]', 'zone_tok\n[3×6 dim]']

# ── Role highlight boxes inside input tokens ──────────────────
def role_highlight(ax, cx, y, tok_label, role_dim_label):
    bw = W_NORM
    bx = cx - bw/2
    box(ax, bx, y, bw, BH, tok_label, C_INPUT, fontsize=7)
    # small yellow highlight for role dims
    hw = bw * 0.32
    hx = bx + bw - hw - 0.05
    hy = y + BH*0.18
    rect = FancyBboxPatch((hx, hy), hw, BH*0.62,
                          boxstyle="round,pad=0.02",
                          linewidth=1, edgecolor='#B5835A',
                          facecolor=C_ROLE, zorder=4)
    ax.add_patch(rect)
    ax.text(hx + hw/2, hy + BH*0.31, role_dim_label,
            ha='center', va='center', fontsize=5.5, color='#7A4100', zorder=5)

# ── Input token layer ─────────────────────────────────────────
role_highlight(ax, xs_center[0], Y_OBS, 'self_tok\n[8]', '[4:7]\nrole')
role_highlight(ax, xs_center[1], Y_OBS, 'ally_tok\n[2×8]', '[5:8]\nrole')
box(ax, xs_center[2]-W_NORM/2, Y_OBS, W_NORM, BH, 'enemy_tok\n[2×8]', C_INPUT, fontsize=7)
box(ax, xs_center[3]-W_NORM/2, Y_OBS, W_NORM, BH, 'zone_tok\n[3×6]', C_INPUT, fontsize=7)

# ── Linear encoders ───────────────────────────────────────────
enc_labels = ['Linear\n8→64', 'Linear\n8→64', 'Linear\n8→64', 'Linear\n6→64']
for i, (cx, lbl) in enumerate(zip(xs_center, enc_labels)):
    box(ax, cx-W_NORM/2, Y_ENC, W_NORM, BH, lbl, C_ENC, fontsize=7)
    arrow(ax, cx, Y_OBS, cx, Y_ENC + BH)

# ── Intra-Set Self-Attention (ally, enemy, zone only) ─────────
sa_xs = [xs_center[1], xs_center[2], xs_center[3]]
sa_labels = ['Ally\nIntra-Set\nSelf-Attn (2H)', 'Enemy\nIntra-Set\nSelf-Attn (2H)',
             'Zone\nIntra-Set\nSelf-Attn (2H)']
for cx, lbl in zip(sa_xs, sa_labels):
    box(ax, cx-W_NORM/2, Y_SA, W_NORM, BH, lbl, C_SELF, fontsize=6.5)
    arrow(ax, cx, Y_ENC, cx, Y_SA + BH)

# self_tok enc goes straight past SA row (no self-attn block)
ax.annotate('', xy=(xs_center[0], Y_CA + BH),
            xytext=(xs_center[0], Y_ENC),
            arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.2,
                            connectionstyle='arc3,rad=0.0'), zorder=5)

# ── Cross-Attention (self as Q, each set as K/V) ──────────────
ca_xs  = [xs_center[1], xs_center[2], xs_center[3]]
ca_lbls = ['Ally\nCross-Attn\n(Q=self, K/V=ally)',
           'Enemy\nCross-Attn\n(Q=self, K/V=enemy)',
           'Zone\nCross-Attn\n(Q=self, K/V=zone)']

for cx, lbl in zip(ca_xs, ca_lbls):
    box(ax, cx-W_NORM/2, Y_CA, W_NORM, BH, lbl, C_CROSS, fontsize=6)
    # arrow from self-attn to cross-attn
    arrow(ax, cx, Y_SA, cx, Y_CA + BH)
    # dashed arrow from self enc to cross-attn (Q path)
    ax.annotate('', xy=(cx - W_NORM/2, Y_CA + BH/2),
                xytext=(xs_center[0], Y_CA + BH/2),
                arrowprops=dict(arrowstyle='->', color='#888888',
                                lw=0.9, linestyle='dashed',
                                connectionstyle='arc3,rad=0.0'), zorder=5)

# Q-label on first dashed arrow
ax.text((xs_center[0] + xs_center[1] - W_NORM/2)/2, Y_CA + BH/2 + 0.08,
        'Q', fontsize=7, color='#555555', ha='center')

# ── Role-routing annotation arrows ────────────────────────────
# Arrow: role in self_tok → Ally Cross-Attn (Q path, highlighted)
ax.annotate('', xy=(xs_center[1] - W_NORM/2 + 0.1, Y_CA + BH),
            xytext=(xs_center[0] - 0.25, Y_OBS),
            arrowprops=dict(arrowstyle='->', color=C_ROLE_ARROW, lw=1.5,
                            connectionstyle='arc3,rad=-0.35'), zorder=6)
ax.text(0.5, 5.6, 'role→Q', fontsize=7, color=C_ROLE_ARROW, fontweight='bold', rotation=70)

# Arrow: role in ally_tok → Ally Cross-Attn (K path, highlighted)
ax.annotate('', xy=(xs_center[1] - W_NORM/2 + 0.55, Y_CA + BH),
            xytext=(xs_center[1] + 0.55, Y_OBS),
            arrowprops=dict(arrowstyle='->', color=C_ROLE_ARROW, lw=1.5,
                            connectionstyle='arc3,rad=0.25'), zorder=6)
ax.text(xs_center[1] + 0.6, 6.25, 'role→K', fontsize=7, color=C_ROLE_ARROW, fontweight='bold')

# ── Output MLP ────────────────────────────────────────────────
mlp_cx = (xs_center[1] + xs_center[3]) / 2
mlp_w  = 3.2
box(ax, mlp_cx - mlp_w/2, Y_MLP, mlp_w, BH,
    'Output MLP: Linear(64×4, 128) → ReLU → Linear(128, action_dim)',
    C_MLP, fontsize=7.5, bold=True)

# arrows from cross-attn outputs to MLP
for cx in ca_xs:
    ax.annotate('', xy=(mlp_cx, Y_MLP + BH),
                xytext=(cx, Y_CA),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=0.9,
                                connectionstyle='arc3,rad=0.0'), zorder=5)

# arrow from self_enc to MLP (direct path)
ax.annotate('', xy=(mlp_cx - mlp_w/2 + 0.2, Y_MLP + BH/2),
            xytext=(xs_center[0], Y_CA + BH),
            arrowprops=dict(arrowstyle='->', color='#888888', lw=0.9,
                            linestyle='dashed',
                            connectionstyle='arc3,rad=0.15'), zorder=5)

# ── Action output ─────────────────────────────────────────────
ax.text(mlp_cx, Y_MLP - 0.25, 'action distribution  (dim = 6)',
        ha='center', va='top', fontsize=8, color='#333333')
arrow(ax, mlp_cx, Y_MLP, mlp_cx, Y_MLP - 0.15)

# ── Legend ────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor='#333', label='Input token'),
    mpatches.Patch(facecolor=C_ENC,    edgecolor='#333', label='Linear encoder'),
    mpatches.Patch(facecolor=C_SELF,   edgecolor='#333', label='Intra-Set Self-Attn + LN'),
    mpatches.Patch(facecolor=C_CROSS,  edgecolor='#333', label='Cross-Attention'),
    mpatches.Patch(facecolor=C_MLP,    edgecolor='#333', label='Output MLP'),
    mpatches.Patch(facecolor=C_ROLE,   edgecolor='#B5835A', label='Role one-hot dims'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=7,
          framealpha=0.9, ncol=3, bbox_to_anchor=(0.0, 0.0))

# ── Title ─────────────────────────────────────────────────────
ax.set_title('SLIC Architecture — Condition 1 (Full)\n'
             'Role one-hots in self_tok and ally_tok condition Q and K/V simultaneously',
             fontsize=10, pad=6)

plt.tight_layout()
out_path = 'D:/Github/SLIC/de_paper/paper/figures/slic_architecture.png'
plt.savefig(out_path, dpi=180, bbox_inches='tight')
print(f"Saved: {out_path}")
