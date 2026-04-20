"""Generate SLIC architecture diagram (Condition 1 — Full) for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Palette ───────────────────────────────────────────────────────────────────
C_INPUT      = '#EDF6FF'
C_ENC        = '#BDE0FE'
C_SELF       = '#8ECAE6'
C_CROSS      = '#FFB3C6'
C_MLP        = '#FF85A1'
C_ROLE       = '#FFD166'
C_ARROW      = '#444444'
C_ROLE_ARROW = '#C1121F'
C_Q          = '#023E8A'
C_KV         = '#6A4C93'

BW = 3.0   # standard box width
BH = 0.82  # standard box height

# Column centers
XS = 2.2, 6.5, 11.0, 15.5   # self, ally, enemy, zone
X_SELF, X_ALLY, X_ENEMY, X_ZONE = XS

# Row y-bottoms  (top-of-stack = y + BH)
Y_IN  = 9.5
Y_ENC = 7.8
Y_SA  = 5.9
Y_CA  = 4.0
Y_MLP = 1.9


# ── Helper: rounded box with label ───────────────────────────────────────────
def box(cx, y, label, color, *, w=BW, h=BH, fs=8, bold=False, lc='#333333', lw=1.3):
    r = FancyBboxPatch((cx - w/2, y), w, h,
                       boxstyle="round,pad=0.06",
                       linewidth=lw, edgecolor=lc, facecolor=color, zorder=3)
    ax.add_patch(r)
    ax.text(cx, y + h/2, label, ha='center', va='center',
            fontsize=fs, fontweight='bold' if bold else 'normal',
            multialignment='center', zorder=4)


# ── Helper: straight arrow ────────────────────────────────────────────────────
def arr(x0, y0, x1, y1, color=C_ARROW, lw=1.3, rad=0.0, style='->',
        lbl=None, lbl_offset=(0.07, 0.07), lbl_fs=6.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'), zorder=5)
    if lbl:
        mx, my = (x0 + x1) / 2 + lbl_offset[0], (y0 + y1) / 2 + lbl_offset[1]
        ax.text(mx, my, lbl, fontsize=lbl_fs, color=color, zorder=6)


# ── Helper: role-dim inset inside an input-token box ─────────────────────────
def token_with_role(cx, y, main_label, role_label):
    box(cx, y, main_label, C_INPUT, fs=7.5)
    hw = BW * 0.26
    hx = cx + BW/2 - hw - 0.08
    hy = y + BH * 0.15
    rp = FancyBboxPatch((hx, hy), hw, BH * 0.70,
                        boxstyle="round,pad=0.03",
                        linewidth=1, edgecolor='#A06020', facecolor=C_ROLE, zorder=4)
    ax.add_patch(rp)
    ax.text(hx + hw/2, hy + BH * 0.35, role_label,
            ha='center', va='center', fontsize=5.5, color='#5A3000', zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# ROW 1  —  Input token groups
# ══════════════════════════════════════════════════════════════════════════════
token_with_role(X_SELF,  Y_IN,  'self_tok  [8-dim]\npos(2), hp, shield, role(3)', '[4:7]\nrole')
token_with_role(X_ALLY,  Y_IN,  'ally_tok  [2×8-dim]\nrel_pos(2), hp, shield, role(3), alive', '[5:8]\nrole')
box(X_ENEMY, Y_IN, 'enemy_tok  [2×8-dim]\nrel_pos(2), hp, shield, type_conf(3), alive', C_INPUT, fs=7.5)
box(X_ZONE,  Y_IN, 'zone_tok  [3×6-dim]\nrel_pos(2), owned, contest, value, pad', C_INPUT, fs=7.5)

# column headers
for cx, lbl in zip(XS, ['self', 'ally', 'enemy', 'zone']):
    ax.text(cx, Y_IN + BH + 0.12, lbl, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#222222')

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2  —  Linear encoders
# ══════════════════════════════════════════════════════════════════════════════
enc_labels = ['Linear  8 → 64', 'Linear  8 → 64', 'Linear  8 → 64', 'Linear  6 → 64']
for cx, lbl in zip(XS, enc_labels):
    box(cx, Y_ENC, lbl, C_ENC, fs=7.5)
    arr(cx, Y_IN, cx, Y_ENC + BH)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3  —  Intra-Set Self-Attention  (ally · enemy · zone only)
# ══════════════════════════════════════════════════════════════════════════════
SA_XS = (X_ALLY, X_ENEMY, X_ZONE)
for cx in SA_XS:
    box(cx, Y_SA, 'Intra-Set\nSelf-Attention  (2H)\n+ Residual + LayerNorm',
        C_SELF, fs=7)
    arr(cx, Y_ENC, cx, Y_SA + BH)

# self_tok enc bypasses self-attn row via a vertical line on the left
arr(X_SELF, Y_ENC, X_SELF, Y_CA + BH, color='#666666', lw=1.1)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4  —  Cross-Attention  (Q = self-emb,  K/V = set)
# ══════════════════════════════════════════════════════════════════════════════
CA_LABELS = [
    'Ally Cross-Attention\n(Q = self_emb,  K/V = ally)',
    'Enemy Cross-Attention\n(Q = self_emb,  K/V = enemy)',
    'Zone Cross-Attention\n(Q = self_emb,  K/V = zone)',
]
for cx, lbl in zip(SA_XS, CA_LABELS):
    box(cx, Y_CA, lbl, C_CROSS, fs=7)
    arr(cx, Y_SA, cx, Y_CA + BH)

# Q-broadcast: horizontal bus from self_enc down to Y_CA midpoint
Q_Y = Y_CA + BH / 2          # y of Q-bus
Q_X0 = X_SELF                 # origin on self column
Q_X_END = X_ZONE              # rightmost cross-attn

# draw the horizontal Q bus line (no arrow head)
ax.plot([Q_X0, Q_X_END + BW/2], [Q_Y, Q_Y],
        color=C_Q, lw=1.4, linestyle='--', zorder=4)

# arrowhead into each cross-attn left edge
for cx in SA_XS:
    ax.annotate('', xy=(cx - BW/2, Q_Y), xytext=(cx - BW/2 - 0.01, Q_Y),
                arrowprops=dict(arrowstyle='->', color=C_Q, lw=1.4), zorder=5)

ax.text(Q_X0 - 0.12, Q_Y + 0.13, 'Q', fontsize=9, color=C_Q,
        fontweight='bold', zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5  —  Output MLP
# ══════════════════════════════════════════════════════════════════════════════
MLP_CX = (X_ALLY + X_ZONE) / 2
MLP_W  = X_ZONE - X_ALLY + BW
MLP_LABEL = ('Concat  [ c_ally | c_enemy | c_zone | e_self ]   (4 × 64 = 256)\n'
             'MLP:  Linear(256, 128) → ReLU → Linear(128, action_dim)')
box(MLP_CX, Y_MLP, MLP_LABEL, C_MLP, w=MLP_W, bold=True, fs=8)

for cx in SA_XS:
    arr(cx, Y_CA, MLP_CX, Y_MLP + BH, color=C_ARROW, lw=1.0, rad=0.0)

# self_emb direct to MLP
ax.annotate('', xy=(MLP_CX - MLP_W/2 + 0.3, Y_MLP + BH/2),
            xytext=(X_SELF, Y_CA + BH),
            arrowprops=dict(arrowstyle='->', color='#666666', lw=1.0,
                            linestyle='dashed',
                            connectionstyle='arc3,rad=0.12'), zorder=5)
ax.text(X_SELF - 0.45, (Y_CA + BH + Y_MLP + BH/2) / 2 + 0.1,
        'e_self', fontsize=6.5, color='#555555')

# Action output
arr(MLP_CX, Y_MLP, MLP_CX, Y_MLP - 0.45)
ax.text(MLP_CX, Y_MLP - 0.55, 'action distribution  (dim = 6)',
        ha='center', va='top', fontsize=8.5, color='#222222')

# ══════════════════════════════════════════════════════════════════════════════
# Role-routing annotation arrows  (RED)
# ══════════════════════════════════════════════════════════════════════════════
# role in self_tok ──▶ Q path
# Arc from role inset in self_tok rightward then down into the Linear encoder
ax.annotate('', xy=(X_SELF - BW/2 - 0.15, Y_ENC + BH * 0.6),
            xytext=(X_SELF + BW/2 - 0.58, Y_IN),
            arrowprops=dict(arrowstyle='->', color=C_ROLE_ARROW, lw=2.0,
                            connectionstyle='arc3,rad=-0.5'), zorder=7)
ax.text(X_SELF - BW/2 - 1.05, (Y_IN + Y_ENC + BH * 0.6) / 2,
        'role→Q', fontsize=8.5, color=C_ROLE_ARROW, fontweight='bold',
        ha='center', va='center')

# role in ally_tok ──▶ K/V path
ax.annotate('', xy=(X_ALLY + BW/2 + 0.2, Y_SA + BH * 0.5),
            xytext=(X_ALLY + BW/2 - 0.58, Y_IN),
            arrowprops=dict(arrowstyle='->', color=C_KV, lw=2.0,
                            connectionstyle='arc3,rad=0.5'), zorder=7)
ax.text(X_ALLY + BW/2 + 0.85, (Y_IN + Y_SA + BH * 0.5) / 2,
        'role→K/V', fontsize=8.5, color=C_KV, fontweight='bold',
        ha='center', va='center')

# ══════════════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════════════
legend_handles = [
    mpatches.Patch(facecolor=C_INPUT,  edgecolor='#555', label='Input token group'),
    mpatches.Patch(facecolor=C_ENC,    edgecolor='#555', label='Linear encoder'),
    mpatches.Patch(facecolor=C_SELF,   edgecolor='#555', label='Intra-Set Self-Attn + LN'),
    mpatches.Patch(facecolor=C_CROSS,  edgecolor='#555', label='Cross-Attention'),
    mpatches.Patch(facecolor=C_MLP,    edgecolor='#555', label='Concat + Output MLP'),
    mpatches.Patch(facecolor=C_ROLE,   edgecolor='#A06020', label='Role one-hot dims'),
    mpatches.Patch(facecolor=C_ROLE_ARROW, edgecolor='#555', label='role → Q  (routing path)'),
    mpatches.Patch(facecolor=C_KV,     edgecolor='#555', label='role → K/V  (routing path)'),
]
ax.legend(handles=legend_handles, loc='lower left', fontsize=7.5,
          framealpha=0.95, ncol=4, bbox_to_anchor=(0.0, -0.01),
          edgecolor='#888888')

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
ax.set_title(
    'SLIC Architecture — Condition 1 (Full)\n'
    'Role one-hots in self_tok [4:7] and ally_tok [5:8] condition Q and K/V simultaneously, '
    'enabling structured role-pair routing',
    fontsize=10.5, pad=10, color='#111111'
)

plt.tight_layout(pad=0.5)
OUT = 'D:/Github/SLIC/de_paper/paper/figures/slic_architecture.png'
plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
