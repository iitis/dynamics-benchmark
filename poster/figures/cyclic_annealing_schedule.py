import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
ORANGE = (230/255, 120/255, 23/255)
GRAY   = (131/255, 130/255, 128/255)

tau_down = 1.4
tau_p    = 0.5
tau_up   = 1.6
cycle    = tau_down + tau_p + tau_up
s_p      = 0.28
def _setup_style(fontsize=15,scale=1.0,grid=True):
# Publication-quality (TikZ-like) matplotlib style setup
        fig_width_pt = 246.0  # width in pt (e.g., for single-column in a paper)
        inches_per_pt = 1.0 / 72.27
        golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # aesthetic ratio
        fig_width = fig_width_pt * inches_per_pt * scale
        fig_height = fig_width * golden_mean
        fig_size = [fig_width, fig_height]
        eps_with_latex = {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": [],
            "font.sans-serif": [],
            "font.monospace": [],
            "axes.labelsize": fontsize,
            "font.size": fontsize,
            "legend.fontsize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "figure.figsize": fig_size,
            "axes.titlesize": fontsize,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "grid.linewidth": 0.5,
            "xtick.direction": 'in',
            "ytick.direction": 'in',
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": grid,
            "grid.alpha": 0.3,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "legend.edgecolor": 'black',
        }
        mpl.rcParams.update(eps_with_latex)

def schedule(t):
    t = t % cycle
    if t < tau_down:
        return 1 - (1 - s_p) * t / tau_down
    elif t < tau_down + tau_p:
        return s_p
    else:
        return s_p + (1 - s_p) * (t - tau_down - tau_p) / tau_up
_setup_style()
n_cycles = 2
t_end = n_cycles * cycle
t = np.linspace(0, t_end, 2000)
s = np.array([schedule(ti) for ti in t])

fig, ax = plt.subplots()

ax.plot(t, s, color=ORANGE, linewidth=2.5)

ax.axhline(1,   color=GRAY, linewidth=0.8, linestyle='--', alpha=0.7)
ax.axhline(s_p, color=GRAY, linewidth=0.8, linestyle='--', alpha=0.7)

# Annotate first cycle
y_arrow = -0.13
arrowprops = dict(arrowstyle='<->', color=GRAY, lw=1.0)
for x0, x1, label in [
    (0,             tau_down,            r'$\tau_{\downarrow}$'),
    (tau_down,      tau_down + tau_p,    r'$\tau_{\mathrm{p}}$'),
    (tau_down+tau_p, cycle,              r'$\tau_{\uparrow}$'),
]:
    ax.annotate('', xy=(x1, y_arrow), xytext=(x0, y_arrow),
                arrowprops=arrowprops)
    ax.text((x0 + x1) / 2, y_arrow - 0.08, label,
            ha='center', va='top', color=GRAY)

# Cycle braces
y_brace = 1.10
for i in range(n_cycles):
    x0, x1 = i * cycle, (i + 1) * cycle
    ax.annotate('', xy=(x1, y_brace), xytext=(x0, y_brace),
                arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=1.0, alpha=0.7))
    ax.text((x0 + x1) / 2, y_brace + 0.03, 'cycle',
            ha='center', va='bottom', color=ORANGE, alpha=0.9)

ax.set_yticks([0, s_p, 1])
ax.set_yticklabels(['$0$', r'$s_{\mathrm{p}}$', '$1$'], color=GRAY)
ax.set_xticks([])
ax.set_xlim(-0.15, t_end + 0.15)
ax.set_ylim(-0.50, 1.45)
ax.set_xlabel('$t$', color=GRAY)
ax.set_ylabel('$s(t)$', color=GRAY)
ax.tick_params(colors=GRAY)
for spine in ax.spines.values():
    spine.set_edgecolor(GRAY)

fig.tight_layout()
fig.savefig('figures/cyclic_annealing_schedule.pdf', bbox_inches='tight')
print("Saved cyclic_annealing_schedule.pdf")
