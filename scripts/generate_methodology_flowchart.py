"""Generate the general methodology flowchart (SVG, vector PDF, 600-dpi PNG).

Run from any directory: python scripts/generate_methodology_flowchart.py
Requires matplotlib.
All figure content is explicit below; no analysis outputs are modified.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = ROOT / 'outputs' / 'figures' / 'methodology'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False,
})
fig = plt.figure(figsize=(12, 9.9), facecolor='white')
# Draw the perimeter in figure coordinates, with its outer edge flush to the
# canvas so exports have a black outline without an exterior white margin.
border_width = 1.0
inset_x = border_width / (2 * 72 * fig.get_figwidth())
inset_y = border_width / (2 * 72 * fig.get_figheight())
fig.add_artist(Rectangle(
    (inset_x, inset_y), 1 - 2 * inset_x, 1 - 2 * inset_y,
    transform=fig.transFigure, fill=False, edgecolor='black',
    linewidth=border_width, clip_on=False, zorder=100,
))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 120), ylim=(-4, 95))
ax.set_axis_off()
INK, ACCENT, PALE = '#202830', '#31536a', 'white'
checks = []

def label(x, y, s, size=11, weight='normal', color=INK, ha='center'):
    return ax.text(x, y, s, ha=ha, va='center', fontsize=size,
                   fontweight=weight, color=color, linespacing=1.3)

def box(x, y, w, h, title, lines=(), dashed=False, fill='white', size=11):
    rect = Rectangle((x, y), w, h, linewidth=0.85, edgecolor=INK,
                     facecolor=fill, linestyle=(0, (4, 3)) if dashed else '-')
    ax.add_patch(rect)
    texts = [label(x+w/2, y+h-1.65, title, size=11.3, weight='bold')]
    if lines:
        top, bottom = y+h-4.1, y+1.55
        ys = [(top+bottom)/2] if len(lines)==1 else [top-(top-bottom)*i/(len(lines)-1) for i in range(len(lines))]
        texts += [label(x+w/2, yy, s, size=size) for yy,s in zip(ys,lines)]
    checks.append((title, rect, texts))
    return rect

def arrow(points, dashed=False):
    style = (0, (3, 2)) if dashed else '-'
    for a,b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=INK,lw=0.85,ls=style)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
        mutation_scale=9, linewidth=0.85, linestyle=style, color=INK,
        shrinkA=0, shrinkB=1.2))

box(14,84,92,9,'CLIMATE PROJECTIONS / ENVIRONMENTAL COVARIATES',[
    'Temperature, humidity, precipitation, wind and atmospheric composition',
    'Select the covariates relevant to the hazard and deterioration mechanism',
],fill=PALE,size=11.5)
arrow([(36,84),(36,82),(19,82),(19,78)])
arrow([(85,84),(85,78)])
for x, title in [(19,'A  NON-STATIONARY LOADING'), (79,'B  CLIMATE-SENSITIVE DETERIORATION')]:
    txt = label(x,79.8,title,11.6,'bold',ACCENT)
    txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=2))

box(2,68,34,10,'Extreme environmental loading',[
    'Extract extremes or load events',
    'Characterize event times and magnitudes',
    'Relate hazard evolution to climate inputs',
],size=10.8)
arrow([(19,68),(19,66)])
box(2,56,34,10,'Alternative loading models',[
    'Non-stationary extreme-value models',
    'NHPP: time-dependent occurrence',
    'LEYP: occurrence with history effects',
],dashed=True,size=10.8)
arrow([(19,56),(19,54)])
box(2,44,34,10,'Statistical estimation and selection',[
    'Estimate time-varying model parameters',
    'Assess event-history dependence',
    'Check fit and select the loading model',
],size=10.8)
arrow([(19,44),(19,42)])
box(2,32,34,10,'Stochastic load process',[
    'Evolving frequency and intensity',
    'Time-dependent load magnitudes',
    r'Total load effect $S(t)$',
],fill=PALE,size=11)

box(40,69,78,9,'Environmental exposure + component properties',[
    'Climate histories, exposure conditions and material characteristics',
    'Geometry and structural details inform the mechanistic deterioration model',
],size=11.5)
arrow([(52,69),(52,67)])
box(40,55,24,12,'Deterioration initiation',[
    'Climate-sensitive mechanism',
    'Onset criterion',
    r'Random initiation time $T_i$',
],size=10.8)
arrow([(64,61),(67,61)])
box(67,55,24,12,'Deterioration propagation',[
    'Environment-dependent kinetics',
    'Damage accumulation',
    'Progressive material loss',
],size=10.4)
arrow([(91,61),(94,61)])
box(94,55,24,12,'Structural capacity loss',[
    'Translate damage into capacity',
    'Mechanistic resistance histories',
    'Normalize by initial capacity',
],size=10.4)
arrow([(106,55),(106,53)])
box(40,46,78,7,'Calibrate stochastic degradation models',[
    'Use mechanistic histories to estimate initiation and climate-dependent propagation parameters',
],fill=PALE,size=11)
arrow([(53,46),(53,44)])
arrow([(82,46),(82,44)])
box(40,32,27,12,'Shifted random-rate model',[
    'Random initiation and rates',
    'Climate-adjusted propagation',
    'Conditionally deterministic paths',
],fill=PALE,size=10.5)
box(69,32,27,12,'Shifted gamma process',[
    'Initiation and positive increments',
    'Climate-dependent shape function',
    'Pathwise stochastic degradation',
],fill=PALE,size=10.5)
box(99,32,19,12,'Initial resistance',[
    'Component properties',
    'Resistance uncertainty',
    r'Sample $R_0$',
],size=10.5)
arrow([(53,32),(53,30)])
arrow([(82,32),(82,30)])
arrow([(108,32),(108,30)])
box(40,25,78,5,'TIME-DEPENDENT STOCHASTIC RESISTANCE: '+r'$R_i(t)=R_{0i}G_i(t)$',[
    r'$G_i(t)=1$ before initiation; climate-sensitive deterioration reduces resistance thereafter',
],fill=PALE,size=10.5)
arrow([(36,37),(38,37),(38,22)])
arrow([(79,25),(79,22)])

box(2,10,116,12,'SEMI-ANALYTICAL RELIABILITY  |  '+r'$g(t)=R(t)-S(t)$; failure when $g(t)\leq0$',[],fill=PALE)
label(20,17.5,'1  Simulate resistance paths',11,'bold')
label(20,14.5,r'$R_j(t),\quad j=1,\ldots,N$',11.5)
label(20,12,'Sample structural uncertainties',10.5)
arrow([(37,15),(41,15)])
label(66,17.5,'2  Evaluate conditional reliability',11,'bold')
label(66,14.5,'Analytical load-process formulation for each path',11)
label(66,12,'No explicit simulation of load-event histories',10.7)
arrow([(91,15),(94,15)])
label(106,17.5,'3  Average paths',11,'bold')
label(106,14.3,r'$\widehat{\mathrm{Rel}}=\frac{1}{N}\sum_j q_j$',12)
label(106,11.8,'Conditional survival '+r'$q_j$',10)

arrow([(60,10),(60,8)])
box(24.5,-2,71,10,'RELIABILITY OUTPUTS AND SERVICE LIFE',[
    r'$P_f(t)=1-\mathrm{Rel}(t),\qquad \beta(t)=-\Phi^{-1}[P_f(t)]$',
    'Service life: first crossing of the adopted target reliability index',
    'Contributions of changing loading, deterioration and combined climate effects',
],size=11)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
overflow = []
for title, rect, texts in checks:
    bb = rect.get_window_extent(renderer)
    for t in texts:
        tb = t.get_window_extent(renderer)
        if tb.x0 < bb.x0+2 or tb.x1 > bb.x1-2 or tb.y0 < bb.y0+1 or tb.y1 > bb.y1-1:
            overflow.append(f'{title}: {t.get_text()}')
    for i, first in enumerate(texts):
        for second in texts[i+1:]:
            if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                overflow.append(f'{title}: overlapping text lines')
if overflow:
    raise RuntimeError('Text exceeds its box: '+repr(overflow))
for ext in ('svg','pdf','png'):
    fig.savefig(OUT / f'methodology_flowchart.{ext}', dpi=600, facecolor='white')
fig.savefig(OUT / 'methodology_flowchart_preview.png', dpi=150, facecolor='white')
plt.close(fig)
guide = '''GENERAL METHODOLOGY FLOWCHART

This revision presents the general framework rather than the illustrative case.
It omits climate-model names, emission-scenario labels, manuscript equation and
section references, fitted magnitude/initial-resistance distributions, numerical
parameters, and the previous annotations below the figure.

Climate projections supply the environmental covariates relevant to each
physical process. They enter two parallel pathways: evolving environmental
loading and climate-sensitive structural deterioration.

The loading pathway extracts extremes/events, considers alternative stochastic
models, estimates their parameters, assesses fit/history effects, and represents
the time-dependent load effect S(t). Dashed model boxes indicate alternatives.

The deterioration pathway combines exposure and component information in a
mechanistic model of initiation, propagation, and capacity loss. The resulting
histories calibrate shifted random-rate and gamma-process models. These are
alternative degradation model classes, not prescribed load-magnitude or initial-
resistance distributions. Random-rate paths are deterministic conditional on
their sampled parameters; gamma-process paths retain increment randomness.

An uncertain initial resistance combines with normalized resistance histories:
R(t)=R0*G(t). The paths remain undegraded before initiation. Initial resistance
calibration is a separate input; no unimplemented geometric conversion is implied.

The branches converge at g(t)=R(t)-S(t). The semi-analytical calculation simulates
resistance paths and averages their conditional survival probabilities q_j,
evaluated with an analytical loading formulation. Load-event histories need not
be explicitly simulated. This is the manuscript's analytically tractable loading
framework, not a claim of a closed form for every possible hazard process.

The outputs are centered beneath the reliability calculation. No separate
climate-effect decomposition panel is included.

Outputs are failure probability, reliability index, service life at the adopted
target crossing, and contributions from the two climate pathways. Interval and
cumulative probabilities must be distinguished in any application. Climate
effects are combined through reliability; no joint stochastic dependence is
introduced by drawing both pathways from the same climate input.

FILES AND REPRODUCTION
SVG: editable vector shapes and text. PDF: vector content with embedded fonts.
PNG: 600 dpi, 7200 x 5940 pixels. Preview: 1800 x 1485 pixels.
Source: scripts/generate_methodology_flowchart.py
Run: python scripts/generate_methodology_flowchart.py
Dependencies are listed in pyproject.toml and requirements.txt.
'''
for name in ('methodology_notes.txt', 'methodology_reading_guide.txt'):
    (OUT / name).write_text(guide, encoding='utf-8')
print(f'Generated SVG, vector PDF, 600-dpi PNG and preview in {OUT}')
