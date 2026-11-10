"""Build the idea ledger mechanically from the round outputs. No text is invented here:
every field is copied from the JSON the idea agents wrote.

Usage, from the repository root:
    python3 docs/research/program/build_ledger.py docs/research/program/raw docs/research/2026-11-09-idea-ledger.md
The raw folder is git-ignored (it names private projects); it exists only on the machine that ran the search."""
import json, sys, glob, os, re
S, OUT = sys.argv[1], sys.argv[2]
def one(s, n=420):
    s = re.sub(r'\s+', ' ', str(s or '')).strip()
    return s if len(s) <= n else s[:n].rsplit(' ', 1)[0] + ' …'
def refs(lst, n=4):
    out = []
    for r in (lst or [])[:n]:
        if isinstance(r, str): out.append(one(r, 160)); continue
        url = r.get('url') or ''
        name = r.get('ref') or r.get('title') or ''
        date = r.get('date') or ''
        read = r.get('opened_and_read', r.get('read_directly'))
        tag = '' if read is None else (' (read)' if read else ' (not read directly)')
        if url.startswith('/'):
            m = re.search(r'/gh_projects/([^/]+)/(.*)$', url)
            # a file on this machine: link it only when it is in this repository
            if m and m.group(1) == 'mac-kernels-public':
                url = os.path.relpath(m.group(2), 'docs/research'); name = name or m.group(2)
            else:
                name = name or ('local notes in ' + (m.group(1) if m else 'another project')); url = ''
        if url and not url.startswith('http') and not url.startswith(('.', '20')) and '/' in url and '.' in url.split('/')[0]: url = 'https://' + url
        label = one(name, 110) if name else url
        out.append((f'[{label}]({url})' if url else label) + (f', {date}' if date else '') + tag)
    return '; '.join(out) if out else 'none listed'
L = []
w = L.append
w('# Idea ledger, 2026-11-09\n')
w('Every idea from the three idea rounds, copied mechanically from the agents\' structured output by a script. '
  'Nothing here is a result. Claims and predictions are hypotheses. "Nearest work" is what the idea\'s author listed; '
  'in rounds 1 and 3 many of those searches were throttled, so treat a missing reference as "not searched", not as "open". '
  'Only the fourteen first-round ideas marked *rechecked* had a full search-backed check. '
  'Long fields are cut at a word boundary and end in an ellipsis. '
  'The reading of all this is in [the research directions document](2026-11-09-research-directions.md).\n')
d = json.load(open(S + '/ideas.json'))
rechecks = {os.path.basename(f)[8:-5]: json.load(open(f)) for f in glob.glob(S + '/r2out/recheck_*.json')}
count = 0
w('## Round 1: nine lenses, 71 ideas\n')
by = {}
for i in d['all']: by.setdefault(i['lens'], []).append(i)
for lens in sorted(by):
    w(f'### Lens: {lens}\n')
    for i in by[lens]:
        count += 1
        w(f"**{one(i['title'],200)}** ({one(i.get('modality'),60)}; self-rated novelty {i.get('self_rated_novelty')})  ")
        w(f"Hypothesis: {one(i.get('hypothesis'))}  ")
        w(f"First experiment: {one(i.get('first_experiment'),300)}  ")
        w(f"Kill criterion: {one(i.get('kill_criterion'),260)}  ")
        w(f"Nearest work listed: {refs(i.get('nearest_prior_art'))}\n")
w('## Round 1 ideas rechecked with real search (round 2)\n')
for c in d['checked']:
    r = rechecks.get(c['id'])
    w(f"### {one(c['title'],200)} (`{c['id']}`)\n")
    w(f"Hypothesis: {one(c.get('hypothesis'))}\n")
    if not r: w('No recheck file.\n'); continue
    sc = r.get('scores') or {}
    w(f"Verdict after search: **{r.get('novelty')}**. Scores (1 to 5): " + ', '.join(f"{k.replace('_',' ')} {v}" for k, v in sc.items())
      + f". Queries run: {len(r.get('queries_run') or [])}; sources opened: {r.get('count_sources_opened')}.\n")
    w('Prior art found:\n')
    for p in (r.get('new_prior_art_found') or []):
        url = p.get('url') or ''; 
        w(f"- [{one(p.get('title'),140)}]({url}), {p.get('date','no date')}. {one(p.get('overlap'),330)}")
    w('')
    w(f"What remains open: {one(r.get('what_remains_open'),900)}\n")
    w(f"Strongest objection: {one(r.get('strongest_objection'),600)}\n")
    w(f"Revised first experiment: {one(r.get('final_first_experiment'),600)}\n")
w('## Round 2: eight new lenses, 48 ideas\n')
for f in sorted(glob.glob(S + '/r2out/ideate2_*.json')):
    j = json.load(open(f)); lens = os.path.basename(f)[8:-5]
    w(f'### Lens: {lens}\n')
    for i in j['ideas']:
        count += 1
        w(f"**{one(i['title'],200)}** ({one(i.get('modality'),60)}; self-rated novelty {i.get('self_rated_novelty')}, ambition {i.get('self_rated_ambition')})  ")
        w(f"Hypothesis: {one(i.get('hypothesis'))}  ")
        w(f"First experiment: {one(i.get('first_experiment'),300)}  ")
        w(f"Kill criterion: {one(i.get('kill_criterion'),260)}  ")
        w(f"Nearest work listed: {refs(i.get('nearest_prior_art'))}\n")
w('## Round 3: twenty-seven lenses, 195 ideas\n')
for f in sorted(glob.glob(S + '/r3out/*.json')):
    j = json.load(open(f)); lens = os.path.basename(f)[:-5]
    w(f'### Lens: {lens}\n')
    for i in j.get('ideas', []):
        count += 1
        w(f"**{one(i['title'],200)}** ({one(i.get('modality'),60)}; wildness {i.get('wildness')})  ")
        w(f"Claim: {one(i.get('claim'))}  ")
        w(f"Prediction: {one(i.get('prediction_with_number'),300)}  ")
        w(f"Control that could embarrass it: {one(i.get('control_that_could_embarrass_it'),260)}  ")
        w(f"Kill criterion: {one(i.get('kill_criterion'),260)}  ")
        w(f"Cost: {one(i.get('cost_estimate'),200)}  ")
        w(f"Nearest work listed: {refs(i.get('closest_known_work'))}  ")
        w(f"Author's note on the search: {one(i.get('prior_art_search_done'),240)}\n")
open(OUT, 'w').write('\n'.join(L) + '\n')
print('ideas', count, 'bytes', os.path.getsize(OUT))
