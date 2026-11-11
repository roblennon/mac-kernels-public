"""Write QUEUE.md from the ordered list below and each card's current title, status line and Needs Rob header.
The order, the reasons and the hand-kept Needs Rob list are the lead's judgment, written here; titles and statuses come
from the cards. Warnings about queued cards go to stdout, not into QUEUE.md."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'designs')
ORDER = [
 ('cross-stack-numerics', 'Extends a probe that already runs in this repository, needs no model, and produces facts every other Mac result in this queue rests on: how Apple GPU and CPU stacks round and add inside a matmul. Citation value is narrow (an Apple column beside published NVIDIA and AMD arithmetic models), but the cross-family reviewer ranked it first for being the most certain to finish and be reused.'),
 ('finetune-survival-requant', "Answers a practical question about Rob's own merge path whatever happens to the law: which fine-tunes survive rounding to 4 bits, and whether one number per layer predicts it against explicit rival predictors. Moved up on the cross-family reviewer's advice."),
 ('noise-margin-gain-table', 'Step 1 builds the instrument that eight separate ideas need (a per-layer noise tolerance with a linearized control), on a small public model. If one calibration prices many shortcuts, format choice becomes a lookup.'),
 ('fixed-vs-fresh-error', "Rob's kernels and formats are the subject. It measures whether 4-bit error behaves as a per-sample bias across sampling steps, the thread seven lenses reached on their own, and either answer is a writable note."),
 ('train-scaffolds', "The card that fits the lab's existing case 001 as it stands; its step 1a is drafted there as Q002, not yet opened. As research it sits lower: its deciding run is days of Mac time and needs Rob's approval."),
 ('rare-feature-threshold', 'A bold law (weight decay and beta2 set the rarest fact a model can keep), cheap to test exactly in a scalar simulation first. Weak case for large models, since the sparse regime is rare in big-batch pretraining.'),
 ('content-energy', 'Whether a token costs different energy depending on what it says, on the Mac. Step 1 is an instrument check other energy work can reuse; later steps have weak priors.'),
 ('rare-exploit-onset', 'CPU-only numpy and decisive for its narrowed optimizer claim. Moved down after the cross-family review found its advertised safety bound was wrong (no exploits in N draws bounds the starting probability at about 3/N, not 1/N squared); the card now claims a finite-regime optimizer result only.'),
 ('sampler-backward-error', 'A reference-free test of fast samplers by inverting their outputs and checking the noise. Cheap on a small public checkpoint; slice 1a settles a real disagreement with published DDIM results.'),
 ('guidance-optimal-weight', 'Forward passes only, with an analytic toy that catches estimator bugs first. Moved down: its distribution-shift reading depends on access to a training-source dataset that is not checked, and the likeliest result is a precise negative that mostly confirms theory.'),
 ('layer-repeat-step-size', 'Separates the two things layer duplication does (a bigger step and a second read) with exact bookkeeping. Cheap, but the headline is judged unlikely; the useful outcome is probably the overshoot account.'),
 ('renyi-search-cutoff', 'A theory-backed second number next to perplexity for search. Rebuilt after a fatal objection into a narrower per-sentence prediction; CPU only.'),
 ('linear-response-window', 'How small a weight nudge must be for training to answer in proportion. No GPU needed, so it can run beside GPU cards; weak carry to large models.'),
 ('continuous-ar-temperature', 'What temperature means in continuous-token generators, by inverting real tokens through the head. Half the idea is occupied by a 2026 paper.'),
 ('surprise-feedback', 'Feed a model its own prediction error as an input. A build, self-contained on synthetic data; the lead\'s prior is low after the review.'),
 ('semantic-latent-memorization', 'Whether a semantic latent lowers the data a diffusion model needs before it stops memorizing. Blocked on Rob accepting the DINOv3 licence.'),
 ('voice-clone-circuit', 'Training-free on a released TTS model: does it clone a voice by looking up matching sounds in the prompt? Active field with two 2026 neighbours, so it comes after the cards above.'),
 ('img-subliminal', 'The long shot. The first step is a CPU-sized MNIST diffusion toy and a positive result would be widely cited, but the cross-family review showed that greater transfer to a same-initialisation student does not by itself separate hidden transfer from ordinary learning of visible changes. The card was revised for that; it moved down until its first slice can identify what it claims.'),
]
# Set aside 2026-11-10 at Rob's direction: behavioural, model-dependent, or not about training, fine-tuning,
# inference, attention, context or gradients. Kept, not deleted; the triage that sorted them is raw/pivot_triage.md.
SET_ASIDE = ['copies-as-evidence', 'agent-hazard-mixture', 'whisper-silence', 'tts-slip-hazard', 'variety-census', 'whisper-grades-students', 'tts-clear-speech-guidance', 'duplex-turn-temperature', 'raster-binding', 'dit-positional-binding', 'denoiser-talks-back']
PARKED = ['paired-metric-floor', 'nested-bitplanes', 'self-training-37']
NOT_CARDS = {'TEMPLATE'}

# What only Rob can give, written by hand from the cards and the reorientation reports of 2026-11-10.
# Once a card carries a "Needs Rob:" header (see designs/TEMPLATE.md), its items are read from the card
# instead; move an item out of this list when its card states it.
NEEDS_ROB = [
 ('semantic-latent-memorization', "Accept the DINOv3 licence on the Hub."),
 ('train-scaffolds', "Approve days of Mac time for step 2, the deciding run."),
 (None, "Whether these research files go public."),
]

# The capability sweep of 2026-11-10 named its gates several ways; these are the names it used.
# New cards use "Subject gate" (designs/TEMPLATE.md).
GATE = re.compile(r'subject gate|subject-property gate|room(-for-effect)? gate|base[- ]task (gate|in reach)|base-rate gate'
                  r'|readability gate|range gate|reach gate|visible-task gate|step 0 gate|capability (step|gate|check)'
                  r'|tests the model, not the instrument', re.I)
STATUS_MAX = 300

def card(cid):
    text = open(os.path.join(D, cid + '.md')).read()
    t = text.splitlines()
    title = t[0].lstrip('# ').strip()
    st = next((l for l in t[:8] if l.startswith('Status')), 'Status: (none)')
    st = re.sub(r'^Status:\s*', '', st)
    needs = []
    for k, l in enumerate(t[:12]):
        if l.startswith('Needs Rob:'):
            rest = l[len('Needs Rob:'):].strip()
            if rest and rest.lower().rstrip('.') != 'nothing':
                needs.append(rest)
            for m in t[k + 1:]:
                if not m.startswith('- '):
                    break
                needs.append(m[2:].strip())
    body = text.split('\n## Review record')[0]
    return title, st, needs, body

def short(st):
    """The status's first sentence or two, at most about STATUS_MAX characters."""
    parts = re.split(r'(?<=[.;])\s+(?=[A-Z"(])', st)
    out = parts[0]
    if len(out) > STATUS_MAX:
        out = out[:STATUS_MAX].rsplit(' ', 1)[0].rstrip(',;:') + '...'
    elif len(parts) > 1 and len(out) + 1 + len(parts[1]) <= STATUS_MAX:
        out += ' ' + parts[1]
    return out if out == st else out + ' (more in the card)'

have = {f[:-3] for f in os.listdir(D) if f.endswith('.md')} - NOT_CARDS
listed = {c for c, _ in ORDER} | set(PARKED) | set(SET_ASIDE)
assert have == listed, (have - listed, listed - have)
cards = {c: card(c) for c in listed}

warnings = []
for cid, _ in ORDER:
    body = cards[cid][3]
    if not re.search(r'^##+ Step 1, exactly', body, re.M):
        warnings.append(f'{cid}: no "Step 1, exactly" section')
    if not GATE.search(body):
        warnings.append(f'{cid}: no subject gate under a recognisable name (rename it "Subject gate")')

need_lines = []
for cid, item in NEEDS_ROB:
    need_lines.append(f'- [{cards[cid][0]}](designs/{cid}.md): {item}' if cid else f'- {item}')
for cid, _ in ORDER:
    for item in cards[cid][2]:
        need_lines.append(f'- [{cards[cid][0]}](designs/{cid}.md): {item}')

L = ['# Experiment queue', '',
 'The experiments worth running first, in order, each with a design card under [designs](designs/). The order was written 2026-11-09 by the lead agent, from 33 cards that each had a real prior-art search, a hostile review and a revision. The card is the source of truth; this file only orders them and says why.', '',
 "Since then, the lab ran card 1's step 1a as its case 002 on 2026-11-10. It decided nothing: every measuring check passed, but the model could not do the base task, so the hypothesis was never tested (the card's \"Result of step 1a\" has the reading). Card 1 was redesigned to start with a capability step, and the other queued cards gained a gate on their subject. What that taught is in [LESSONS.md](LESSONS.md); new cards start from [the template](designs/TEMPLATE.md).", '',
 '## Focus, from 2026-11-10', '',
 "Rob redirected the programme on 2026-11-10 toward training, fine-tuning, inference, attention, context and gradients, and away from behaviour. The behavioural cards are set aside below. The thread now leading is not a card: [decode and small-batch speed on MPS](threads/mps-decode.md), opened from a lab result and measured the same day ([findings](../../findings.md)). The cards below are ordered for that focus; the reasons under each were written for the earlier order and still describe the card.", '',
 '## Needs Rob', '',
 'Nothing below runs past these without him. The lab asks him for its own approvals; this list is so they are visible in one place.', '',
 *need_lines, '',
 '## How the order was chosen', '',
 'Four things, in this order of weight. A first step that runs on the Mac inside the lab\'s limits (30 GPU-minutes, 300 seconds per process, 8 GiB) and already decides something. What a yes would change, and whether it carries to larger models. Whether it sits on ground only Rob has: his kernels and formats, his voice work, his steering tools. And spread, so the top of the queue is not one bet made many times, which is the warning in [the musings](MUSINGS.md#a-sceptical-reader-on-the-whole-notebook).', '',
 'A reviewer from a different model family (GPT-6.1 Sol) read the queue and its top cards cold ([its review](reviews/2026-11-09-queue-review-sol.md)). It found the top of the first draft made one operational bet over and over: a cheap first slice that only showed feasibility, followed by a long road to a citable result. The order below follows most of its advice. Cards whose first slice decides something moved up; cards whose first slice only shows feasibility moved down. The top cards were then repaired against its list and checked by a fresh agent who tried to plan each one cold.', '',
 'The self-rated chances on the cards are low on purpose: most say 2 out of 5. A null written up honestly is a deliverable, and several cards are built to return a tight null.', '',
 'Each card opens with a status line, shortened here. "Search gate" means the card\'s prior-art search had not passed when this was written; run it first, because the card may park itself.', '',
 '## For the lab', '',
 "The lab's case 001 measures charged seconds to a target loss on a small character model, and the lab has already run questions on it. Card 14, removable training scaffolds, fits case 001 as it stands; its step 1a is drafted there as Q002, not yet opened. Case 002 ran card 1's step 1a. Every other card has a non-speed outcome and needs a new case with its own contract, noise floor and planted-effect check; each card describes both checks. Bind the lab's plan to the card's commit, and put the card's predictions into the lab's plan ledger before any run.", '',
 '## The queue', '']
for i, (cid, why) in enumerate(ORDER, 1):
    title, st = cards[cid][:2]
    L += [f'{i}. **[{title}](designs/{cid}.md)**  ', f'   Status: {short(st)}  ', f'   Why here: {why}', '']
L += ['## Set aside (behavioural)', '', 'Set aside on 2026-11-10 at Rob\'s direction. Two hold a speed angle worth salvaging: whisper-silence (if the padding\'s content does not matter, short clips spend encoder compute on up to 30 s of padding) and denoiser-talks-back (if the text stream barely depends on the seed, it can be cached across seeds and steps).', '']
for cid in SET_ASIDE:
    title, st = cards[cid][:2]
    L += [f'- [{title}](designs/{cid}.md)', '']
L += ['## Parked', '', 'Kept because the reasoning is worth having; each card says what would unpark it.', '']
for cid in PARKED:
    title, st = cards[cid][:2]
    L += [f'- **[{title}](designs/{cid}.md)**. {short(st)}', '']
open(os.path.join(HERE, 'QUEUE.md'), 'w').write('\n'.join(L))
print('wrote', len(ORDER), 'queued,', len(SET_ASIDE), 'set aside and', len(PARKED), 'parked')
for w in warnings:
    print('warning:', w)
