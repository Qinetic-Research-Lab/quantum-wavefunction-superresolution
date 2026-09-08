"""Build the ML4PS (NeurIPS-style, 4 pages + references) PDF of the paper.

Usage (from the repository root):

    python paper/make_figure1.py     # writes outputs/analysis/figure1.png
    python paper/build_paper.py      # writes paper/ML4PS_submission.pdf

Every number in the text is taken from the released analysis outputs:
outputs/analysis/summary_canonical.json (seed 42, headline run),
outputs/fourier_baseline_fix_report.json, outputs/oracle_baseline_report.json,
outputs/sigma_space_reanalysis.json, outputs_seed43..46/baseline_summary.json,
outputs_widerange_ablation/baseline_summary.json, outputs/analysis/timing_summary.json.
Fonts: Times New Roman TTFs; set PAPER_FONT_DIR on non-Windows platforms.
"""
import os
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, Image, KeepTogether)

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "outputs" / "analysis" / "figure1.png"
OUT = REPO / "paper" / "ML4PS_submission.pdf"
FONT_DIR = os.environ.get("PAPER_FONT_DIR", r"C:\Windows\Fonts")

pdfmetrics.registerFont(TTFont("NR", os.path.join(FONT_DIR, "times.ttf")))
pdfmetrics.registerFont(TTFont("NB", os.path.join(FONT_DIR, "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("NI", os.path.join(FONT_DIR, "timesi.ttf")))
pdfmetrics.registerFont(TTFont("NBI", os.path.join(FONT_DIR, "timesbi.ttf")))
registerFontFamily("NR", normal="NR", bold="NB", italic="NI", boldItalic="NBI")

TW, TH = 5.5 * inch, 9.0 * inch          # NeurIPS text block
LM = (LETTER[0] - TW) / 2.0
BM = (LETTER[1] - TH) / 2.0

S = dict(
    title=ParagraphStyle("t", fontName="NB", fontSize=14, leading=17,
                         alignment=TA_CENTER, spaceAfter=8),
    anon=ParagraphStyle("a", fontName="NR", fontSize=10, leading=12,
                        alignment=TA_CENTER, spaceAfter=10),
    abshead=ParagraphStyle("ah", fontName="NB", fontSize=11, leading=13,
                           alignment=TA_CENTER, spaceAfter=4),
    abs=ParagraphStyle("ab", fontName="NR", fontSize=10, leading=11.6,
                       alignment=TA_JUSTIFY, leftIndent=24, rightIndent=24,
                       spaceAfter=8),
    h1=ParagraphStyle("h1", fontName="NB", fontSize=12, leading=14,
                      spaceBefore=6, spaceAfter=2),
    body=ParagraphStyle("b", fontName="NR", fontSize=10, leading=11.6,
                        alignment=TA_JUSTIFY, spaceAfter=1.5),
    cap=ParagraphStyle("c", fontName="NR", fontSize=8.6, leading=10.2,
                       alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=4),
    refs=ParagraphStyle("r", fontName="NR", fontSize=8.8, leading=10.4,
                        alignment=TA_JUSTIFY, leftIndent=12, firstLineIndent=-12,
                        spaceAfter=2),
)


def P(t, s="body"):
    return Paragraph(t, S[s])


def page_num(canvas, doc):
    canvas.saveState()
    canvas.setFont("NR", 9)
    canvas.drawCentredString(LETTER[0] / 2.0, BM - 26, str(canvas.getPageNumber()))
    canvas.restoreState()


W = "\u03c9"      # omega
R = "\u03c1"      # rho
SG = "\u03c3"     # sigma
X = "\u00d7"      # times
MINUS = "\u2212"


def e(m, k):      # scientific notation m x 10^-k in markup
    return f"{m} {X} 10<super>{MINUS}{k}</super>"


story = []
story.append(P("Where Classical Interpolation and a Convolutional Surrogate Fail: "
               "Wavefunction Super-Resolution on the 1D Quantum Harmonic Oscillator", "title"))
story.append(P("Anonymous Author(s)<br/>Affiliation withheld for review", "anon"))

story.append(P("Abstract", "abshead"))
story.append(P(
 "We train a one-dimensional convolutional network to reconstruct 1024-point ground-state wavefunctions of the "
 "quantum harmonic oscillator from 64-point coarse solutions, and compare it with cubic-spline interpolation on "
 "200 matched held-out samples. Both methods clear the <i>F</i> &gt; 0.99 fidelity threshold on "
 "every sample (network mean 0.999832, spline 0.999790), so the comparison turns on how each method's error is "
 "organized over the frequency range. The difference is sharpest where the physics is easiest: in the "
 "lowest-frequency bin, where the coarse grid resolves the state best, the spline attains its lowest error "
 "while the network attains its highest, on identical inputs. Spline infidelity is perfectly "
 f"rank-correlated with frequency (Spearman {R} = +1.000), a resolution-limited failure, whereas network "
 f"infidelity follows an inverted-U over the sampled interval, worst at both edges ({R} = +0.861 against "
 "distance from the midpoint). Retraining on a wider interval moved error off the old edges but not "
 "symmetrically onto the new ones, only partly supporting a training-density explanation.", "abs"))

story.append(P("1&nbsp;&nbsp;Introduction", "h1"))
story.append(P(
 "Direct solution of the time-independent Schr\u00f6dinger equation (TISE) on a fine grid scales badly: an "
 f"<i>N</i>-point 1D problem requires diagonalizing an <i>N</i> {X} <i>N</i> Hamiltonian, and dimension and "
 "particle number compound the cost exponentially. A natural mitigation is to solve cheaply on a coarse grid and "
 "reconstruct the fine solution with a learned model, by analogy to image super-resolution. We train a "
 "convolutional network, QuantumResNet, to map a 64-point coarse ground state of the quantum harmonic oscillator "
 "to its 1024-point counterpart, scored by quantum fidelity."))
story.append(P(
 "The 1D harmonic oscillator does not need machine learning; it solves in tens of milliseconds on our hardware. "
 "It is a controlled setting with exact ground truth across a densely sampled parameter family. A cubic spline "
 "clears <i>F</i> &gt; 0.99 across the whole tested range, so a higher mean fidelity for the network would say "
 "little on its own. The informative question is what organizes each method's error, and the answer differs: "
 "the spline's by the physics of undersampling, the network's by position within its training range."))
story.append(P(
 "Three statements bound the claim. The model is a supervised surrogate, not a physics-informed neural network: "
 "its loss is pure mean-squared error against precomputed eigenvectors, with no Schr\u00f6dinger-residual term, "
 "the defining feature of a PINN [1]. \u201cQuantum\u201d here means quantum simulation on classical hardware; "
 "there are no qubits or circuits in this pipeline. The measured speedup is machine-specific and secondary."))

story.append(P("2&nbsp;&nbsp;Related Work", "h1"))
story.append(P(
 "PINNs embed a PDE residual in the training loss [1]; ours does not. Neural operators such as the FNO learn "
 "resolution-independent maps between function spaces with zero-shot super-resolution [2]; ours is "
 "resolution-specific but addresses the same problem of reconstructing fine fields from coarse observations of a "
 f"family indexed by {W}, as in SRCNN [3]. Closest are convolutional "
 "surrogates for the Schr\u00f6dinger equation: Mills et al. [4] map a 2D potential to a ground-state energy, "
 "whereas we reconstruct the eigenvector; Li et al. [5] apply super-resolution-inspired upsampling to electron "
 "densities, scoring by energy error. Neural-network quantum states parameterize the wavefunction variationally "
 "with no precomputed ground truth [6]. That learned surrogates degrade away from their training distribution is "
 "well documented: Mouli et al. [7] report neural operators failing on out-of-domain PDE inputs at high "
 "in-domain accuracy, and Gopakumar et al. [8] construct conformal intervals with marginal coverage across a "
 "surrogate's operating domain under an exchangeability assumption. What this setting adds is separability: a "
 "classical baseline runs on identical inputs and its error is ordered by physics without exception "
 f"({R} = +1.000, zero rank inversions across 200 samples), so the two failure modes can be observed side by "
 "side."))

story.append(P("3&nbsp;&nbsp;Methods", "h1"))
story.append(P(
 "<b>System and ground truth.</b> We solve the 1D TISE in atomic units (\u0127 = <i>m</i> = 1), "
 "\u2212\u00bd\u03c8\u2033 + <i>V</i>(<i>x</i>)\u03c8 = <i>E</i>\u03c8, with <i>V</i>(<i>x</i>) = "
 f"\u00bd{W}<super>2</super><i>x</i><super>2</super> on the domain [{MINUS}10, 10] with Dirichlet boundaries. A "
 "three-point finite-difference stencil yields a sparse Hamiltonian; the ground state comes from "
 "<font face=\"NI\">eigsh(H, k=1, which=\"SA\")</font>, solved independently at <i>N</i> = 1024 (target) and "
 "<i>N</i> = 64 (input). Wavefunctions are <i>L</i><super>2</super>-normalized with a fixed sign convention; "
 "only the nodeless <i>n</i> = 0 state is used, at fixed potential centre <i>x</i><sub>0</sub> = 0. "
 f"Frequency sets physical difficulty quantitatively: the state is a Gaussian of width {SG} = 1/\u221a{W} "
 f"against grid spacing \u0394<i>x</i> = 20/63 \u2248 0.317, so coarse points per width fall from 4.45 at {W} = "
 f"0.5 to 1.41 at {W} = 5.0."))
story.append(P(
 f"<b>Data and training.</b> 2,000 samples, {W} ~ <i>U</i>[0.5, 5.0], seed 42; split 72/18/10 (1,440/360/200). "
 "QuantumResNet maps [<i>B</i>, 1, 64] \u2192 [<i>B</i>, 1, 1024] through four transposed-convolution upsample "
 "stages (kernel 4, stride 2, padding 1), each followed by a pre-activation residual block (BatchNorm \u2192 GELU "
 f"\u2192 Conv1d, twice, with skip), 64 channels throughout; 166,209 parameters. Adam (lr 10<super>{MINUS}3</super>, "
 "batch 64), MSE loss, cosine annealing over the 30-epoch budget; early stopping (patience 7) restored the "
 "best-validation checkpoint. A run takes 113 s on the CPU named in Section 4. Early stopping did not fire, "
 "which is not evidence of headroom: the schedule drives the learning rate to near zero by epoch 30 by "
 "construction. Robustness checks retrain the identical configuration under seeds 43 to 46, and one single-seed "
 f"ablation retrains on {W} in [0.3, 7.0] with <i>N</i> = 3,000 (448 samples per unit {W} against the "
 "canonical 444; split 2,160/540/300)."))
story.append(P(
 "<b>Metrics and baselines.</b> Fidelity is <i>F</i> = (\u03a3<sub>i</sub> \u03c8<sub>hr,i</sub> "
 "\u03c8<sub>pred,i</sub> \u0394<i>x</i>)<super>2</super>, both wavefunctions renormalized under the discrete-sum "
 "rule of the released code; the analysis works with infidelity 1 \u2212 <i>F</i>. Baselines: a cubic spline "
 "(<font face=\"NI\">CubicSpline</font>, 64 \u2192 1024); Fourier zero-padding applied to the 63 unique periodic "
 f"points of the endpoint-inclusive grid; and an oracle-family baseline that estimates {W} from the coarse input "
 f"alone through the ground-state identity var(<i>x</i>) = 1/(2{W}) and emits the analytic ground state "
 f"({W}/\u03c0)<super>1/4</super> exp({MINUS}{W}<i>x</i><super>2</super>/2) on the fine grid. All share the "
 "fidelity function and test slice; none is tuned. Error shapes are summarized by Spearman correlations of "
 f"infidelity against {W} and against |{W} \u2212 {W}<sub>mid</sub>| ({W}<sub>mid</sub> = 2.750). These are "
 "one curve viewed twice: the second predictor is a folded transform of the first, so the pair restates Table "
 "1's shapes and does not test between explanations."))

story.append(P("4&nbsp;&nbsp;Results", "h1"))
story.append(P(
 f"<b>Fidelity.</b> Train loss fell to {e('9.755', 6)} and validation to {e('7.895', 6)} without divergence. "
 "On the 200-sample test set the network reached mean fidelity 0.999832 (median 0.999889, min 0.999328, max "
 f"0.999909, std {e('1.13', 4)}), 0/200 below <i>F</i> = 0.99; the spline reached 0.999790 (median 0.999831, "
 f"min 0.999442, max 0.999995, std {e('1.65', 4)}). Across seeds 42 to 46, network mean fidelity is 0.999869 "
 "(range 0.999832 to 0.999914) and the spline's 0.999793 (range 0.999769 to 0.999818); seed 42, the headline "
 "run throughout, is the lowest of the five. The Fourier baseline as first implemented scored 0.969008. The "
 "cause was a grid-convention error, not a property of Fourier interpolation: the endpoint-inclusive 64-point "
 "array was transformed as if periodic in index space, a rigid shift of 0.149 length units at the domain centre. "
 "Corrected, it scores 0.999770, comparable to the spline (Table 1); an earlier version of this comparison "
 "excluded it on that erroneous basis. The oracle baseline, which knows the "
 "functional family exactly, reaches 0.999789 and tracks the spline bin by bin rather than the network. A "
 f"diagnostic substituting the true {W} gives <i>F</i> = 1.000000 against the numerical target, so the oracle's "
 f"shortfall is entirely its {W} estimate from 64 points (mean |\u0394{W}| = 0.13), not solver discretization: "
 "classical error here is set by what the coarse grid determines about the state."))

hdr = [f"{W} range", "n", "Spline F", "FFT F", "Oracle F", "CNN F",
       "Spline 1\u2212F", "CNN 1\u2212F", "CNN wins"]
rows = [
    ["[0.5, 1.4)", "36", "0.999980", "0.999979", "0.999981", "0.999680", "2.03e\u22125", "3.20e\u22124", "0 / 36"],
    ["[1.4, 2.3)", "36", "0.999932", "0.999930", "0.999936", "0.999894", "6.75e\u22125", "1.06e\u22124", "4 / 36"],
    ["[2.3, 3.2)", "46", "0.999836", "0.999826", "0.999840", "0.999896", "1.64e\u22124", "1.04e\u22124", "46 / 46"],
    ["[3.2, 4.1)", "42", "0.999708", "0.999682", "0.999709", "0.999892", "2.92e\u22124", "1.08e\u22124", "42 / 42"],
    ["[4.1, 5.0]", "40", "0.999527", "0.999466", "0.999511", "0.999777", "4.73e\u22124", "2.23e\u22124", "40 / 40"],
    ["Overall", "200", "0.999790", "0.999770", "0.999789", "0.999832", "2.10e\u22124", "1.68e\u22124", "132 / 200"],
]
tbl = Table([hdr] + rows, hAlign="CENTER",
            colWidths=[0.62*inch, 0.28*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch,
                       0.62*inch, 0.62*inch, 0.6*inch])
tbl.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, 0), "NB"), ("FONTSIZE", (0, 0), (-1, -1), 7.2),
    ("FONTNAME", (0, 1), (-1, -1), "NR"), ("FONTNAME", (0, -1), (-1, -1), "NB"),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LINEABOVE", (0, 0), (-1, 0), 0.7, colors.black),
    ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
    ("LINEABOVE", (0, -1), (-1, -1), 0.4, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
    ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
]))

story.append(P(
 "<b>Matched comparison.</b> The network wins 132 of 200 matched samples (66%; one-sided binomial <i>p</i> = "
 f"{e('3.5', 6)}; Wilcoxon on paired infidelities <i>W</i> = 6588, <i>p</i> = {e('2.4', 5)}), with 1.25{X} "
 "lower mean infidelity overall. Both methods clear <i>F</i> &gt; 0.99 on every sample, so these are differences "
 "in the structure of very small errors, not a pass/fail separation. The aggregate conceals a reversal (Figure "
 "1): the network loses all 36 samples in the lowest bin and 32 of 36 in the second, then wins all 128 in the "
 f"upper three. Below {W} = 1.5 it wins 0 of 42; at {W} \u2265 3.5 it wins 65 of 65 with 2.31{X} lower mean "
 f"infidelity. For this seed the spline's last win falls at {W} = 2.03 and the network wins every sample from "
 f"{W} = 2.09 upward; across the five seeds that unbroken run begins between {W} = 1.73 and 2.10 (mean 1.95) "
 "and the network wins 122 to 157 of 200, so the crossover is a band, not a fixed value."))

story.append(Spacer(1, 2))
story.append(KeepTogether([tbl, P(
    f"Table 1: Fidelity and infidelity by {W} bin; all methods on the same 200 matched samples (seed 42). "
    "FFT is the corrected Fourier baseline.", "cap")]))

img = Image(str(FIG))
img._restrictSize(TW, 1.7 * inch)
story.append(KeepTogether([img, P(
 f"Figure 1: (a) Infidelity against {W} for spline and network on the 200 matched samples (open markers), bin "
 f"means overlaid, log ordinate; the curves cross in the shaded band near {W} \u2248 2. (b) Per-sample ratio "
 "of spline to network infidelity; its crossing of unity locates the same transition.", "cap")]))

story.append(P(
 "<b>Two error structures.</b> The clearest evidence comes from where the physics is easiest. In the lowest bin "
 f"the coarse grid is at its most generous, and the spline attains {e('2.03', 5)}, the lowest error of any "
 f"method in any bin; on those same samples the network attains {e('3.20', 4)}, its own worst and roughly "
 "three times its interior error. Since the spline's difficulty is set entirely by grid resolution, "
 "undersampling cannot be what the network is failing on. Numerically, spline infidelity is perfectly "
 f"rank-correlated with {W} ({R} = +1.000, zero inversions) and uncorrelated with the folded predictor "
 f"({R} = +0.031, <i>p</i> = 0.66); network infidelity is uncorrelated with {W} ({R} = +0.026, <i>p</i> = "
 f"0.71) and strongly correlated with distance from the midpoint ({R} = +0.861, <i>p</i> = {e('4.0', 60)}), "
 f"restating the inverted-U. Re-expressing the predictor as {SG} = 1/\u221a{W} or log {W} leaves these values "
 "unchanged up to sign, as monotone reparametrization must, excluding a parametrization artefact without adding "
 f"evidence. Low-edge infidelity exceeds high-edge by 1.433{X}."))
story.append(P(
 "A distributional reading fits this pattern: accuracy set by local training density, worst where a sample has "
 f"neighbours on one side only. To test it we retrained once on {W} in [0.3, 7.0] at matched density. The first "
 f"prediction held: the former edges, now interior, improved from {e('3.20', 4)} to {e('1.09', 4)} ({W} in "
 f"[0.5, 1.4)) and from {e('2.23', 4)} to {e('5.45', 5)} ({W} in [4.1, 5.0)). The second did not: error "
 f"concentrated at the new low edge ({e('3.15', 4)} for {W} &lt; 0.5, <i>n</i> = 9) but only mildly at the new "
 f"high edge ({e('7.40', 5)} for {W} in [5, 7], against an interior of 5 to 6 {X} 10<super>{MINUS}5</super>). "
 f"Mean fidelity rose to 0.999923 and the crossover moved to {W} = 1.53. Edge error therefore follows the "
 "training boundary, but asymmetrically, which distance to the boundary alone does not explain; Section 5 offers "
 "a candidate reason. The alternative that extremal widths are intrinsically hard for a fixed receptive field "
 "is not excluded either. The ablation is one seed, and its new low bin holds nine samples."))
story.append(P(
 "<b>Timing (machine-specific).</b> Ten repetitions of the pipeline's 50-run benchmark on an Intel Core Ultra 7 "
 "256V (CPU): median 20.6 ms per 1024-point solve (range 17.7\u201324.1) against 4.9 ms for the coarse solve "
 f"plus inference (range 4.7\u20137.1), median speedup 4.1{X} (range 3.1\u20134.9); machine- and load-dependent, "
 "and not evidence of practical acceleration at a tens-of-milliseconds baseline."))

story.append(P("5&nbsp;&nbsp;Discussion", "h1"))
story.append(P(
 "The spline's error is bounded by the information in 64 samples, and the oracle sharpens that: exact knowledge "
 f"of the functional family does not beat the spline, because the coarse grid fixes the width only to about "
 f"0.13 in {W}. The network's error is "
 "instead a function of training-set design, and the ablation shows that moving a boundary moves the error, "
 f"though not symmetrically. A candidate explanation is that the relevant density is not in {W} but in the "
 f"shape the network must reproduce: sampling uniformly in {W} gives a density per unit width {SG} that scales "
 f"as 2/{SG}<super>3</super>, so the broad, low-{W} states are sparsest in shape space and stay sparse after "
 f"the range is widened, while narrow, high-{W} states are densely covered. That would produce the observed "
 "pattern. It is an inference from the pattern, not a tested claim; a direct test would sample uniformly in "
 f"{SG} or log {W}."))
story.append(P(
 "One corollary cuts against the network. A surrogate whose accuracy tracks its training data is weakest exactly "
 "where this class of method is meant to be used, where high-resolution ground truth is scarcest. The concern is "
 "established [7, 8]; what this setting contributes is isolation, since a physics-ordered classical control on "
 "identical inputs makes the network's departure from that ordering directly observable."))

story.append(P("6&nbsp;&nbsp;Limitations", "h1"))
story.append(P(
 f"<b>Scope.</b> Every sample is a harmonic ground state at fixed <i>x</i><sub>0</sub> = 0, with only {W} "
 "randomized; shifted potentials, other functional forms (double-well and anharmonic are implemented but "
 f"untrained), and excited states with nodes are untested. Test {W} values lie inside the training range, so "
 "this characterizes degradation toward the training edges, not extrapolation beyond them."))
story.append(P(
 f"<b>Mechanism attribution.</b> The distributional explanation rests on the low-{W} asymmetry, the error "
 "shape, and one widened-range retrain. That retrain confirmed edge error moves inward when a boundary is moved "
 "outward but found the new edges unequal, so it settles neither the distance-to-boundary account nor the "
 "receptive-field alternative, and raises a third candidate, sampling density in shape space, that was not "
 "tested. It is single-seed and its new low bin holds nine samples. The rank correlations add no independent "
 "support, and the analysis assumes a one-dimensional parameter space in which \u201cinterior\u201d is "
 "unambiguous."))
story.append(P(
 "<b>Seed variance and configuration.</b> Headline numbers come from one model and one split (seed 42). Across "
 f"five seeds, network mean fidelity spans 0.999832 to 0.999914 and the crossover spans {W} = 1.73 to 2.10, so "
 "the crossover values in Section 4 are one realization and the margin over the spline varies with seed; the "
 "inverted-U with a worse low edge appeared in every seed. The architecture was fixed a priori rather than "
 "ablated."))
story.append(P(
 "<b>Fidelity is the only accuracy metric.</b> It is an amplitude-weighted global overlap, insensitive to "
 "high-frequency content and boundary violation, both visible in this model's residuals. Because the kinetic "
 "operator involves a second derivative, a reconstruction scoring well on fidelity may score worse on energy or "
 "on \u2016<i>H</i>\u03c8 \u2212 <i>E</i>\u03c8\u2016; no physics-based metric was computed, and the ranking "
 "here should not be assumed to survive one."))

story.append(P("7&nbsp;&nbsp;Conclusion", "h1"))
story.append(P(
 "The two methods' errors are organized by different variables, most visibly where the physics is easiest. Spline "
 f"error is perfectly rank-ordered by {W}, a resolution-limited failure that an oracle knowing the functional "
 "family shares; network error follows an inverted-U that five seeds reproduce. Widening the training range "
 "moved error off the old edges but not symmetrically onto the new ones, so training density in the shape the "
 "network must produce, rather than in the raw parameter, is the open question. Next steps: a direct test of "
 f"that hypothesis by training uniform in {SG} or log {W}; physics-based evaluation with energy and residual "
 "metrics; and extension to the double-well and anharmonic potentials the solver supports."))
story.append(P(
 "<b>LLM use.</b> LLM assistance was used for drafting, analysis code (including the Fourier correction, oracle "
 "baseline, ablation and seed replication), and auditing of citations and numbers; all results were verified "
 "against the released code. <b>Code and data.</b> An anonymized snapshot of the code, checkpoints, per-sample "
 "metrics and analysis scripts accompanies this submission; the released scripts regenerate every number "
 "reported here."))

story.append(P("References", "h1"))
for r in [
 "[1] Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics-informed neural networks: a deep learning "
 "framework for solving forward and inverse problems involving nonlinear partial differential equations. "
 "<i>Journal of Computational Physics</i>, 378:686\u2013707, 2019.",
 "[2] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. "
 "Fourier neural operator for parametric partial differential equations. In <i>International Conference on "
 "Learning Representations (ICLR)</i>, 2021.",
 "[3] Dong, C., Loy, C. C., He, K., and Tang, X. Image super-resolution using deep convolutional networks. "
 "<i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, 38(2):295\u2013307, 2016.",
 "[4] Mills, K., Spanner, M., and Tamblyn, I. Deep learning and the Schr\u00f6dinger equation. <i>Physical "
 "Review A</i>, 96(4):042113, 2017.",
 "[5] Li, C., Sharir, O., Yuan, S., and Chan, G. K. Image super-resolution inspired electron density "
 "prediction. <i>Nature Communications</i>, 16:4811, 2025.",
 "[6] Carleo, G. and Troyer, M. Solving the quantum many-body problem with artificial neural networks. "
 "<i>Science</i>, 355(6325):602\u2013606, 2017.",
 "[7] Mouli, S. C., Maddix, D. C., Alizadeh, S., Gupta, G., Stuart, A., Mahoney, M. W., and Wang, Y. Using "
 "uncertainty quantification to characterize and improve out-of-domain learning for PDEs. In <i>International "
 "Conference on Machine Learning (ICML)</i>, 2024.",
 "[8] Gopakumar, V., Gray, A., Oskarsson, J., Zanisi, L., Giles, D., Kusner, M. J., Pamela, S., and "
 "Deisenroth, M. P. Uncertainty quantification of surrogate models using conformal prediction. <i>Machine "
 "Learning: Science and Technology</i>, 7(1):015025, 2026.",
]:
    story.append(P(r, "refs"))

doc = BaseDocTemplate(str(OUT), pagesize=LETTER, leftMargin=LM, rightMargin=LM,
                      topMargin=BM, bottomMargin=BM,
                      title="Where Classical Interpolation and a Convolutional Surrogate Fail",
                      author="Anonymous")
frame = Frame(LM, BM, TW, TH, leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0)
doc.addPageTemplates([PageTemplate(id="n", frames=[frame], onPage=page_num)])
doc.build(story)
print("WROTE", OUT)
