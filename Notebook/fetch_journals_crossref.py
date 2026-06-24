"""
Build journal authority from Crossref (fast, no aggressive rate limits).
Step 1: Download all Crossref journals → names + ISSNs
Step 2: Match our proposal journal strings to Crossref by name → get ISSNs
Step 3: Use those ISSNs to fetch OpenAlex source IDs (small batch)
Output: Data/openalex_sources_journals.parquet
"""
import requests, pandas as pd, re, time, json
from pathlib import Path
from tqdm.auto import tqdm
from difflib import SequenceMatcher

WORK_DIR = Path(__file__).parent
DATA_DIR = WORK_DIR.parent / 'Data'
OUT_PATH = DATA_DIR / 'openalex_sources_journals.parquet'
CROSSREF_CACHE = DATA_DIR / 'crossref_journals.parquet'
MAILTO = 'mchalekson@gmail.com'

# ── Step 1: Download Crossref journal list ─────────────────────────────────────
if CROSSREF_CACHE.exists():
    print(f'Loading cached Crossref journals...')
    cr = pd.read_parquet(CROSSREF_CACHE)
    print(f'  {len(cr):,} journals loaded')
else:
    print('Downloading Crossref journal list (~167k journals)...')
    rows = []
    offset = 0
    per_page = 1000
    CHECKPOINT = DATA_DIR / 'crossref_checkpoint.json'
    session = requests.Session()
    session.headers.update({'User-Agent': f'ProposalAtypicality/1.0 (mailto:{MAILTO})'})

    # Get total count first
    r = session.get(f'https://api.crossref.org/journals?rows=1&mailto={MAILTO}', timeout=30)
    total = r.json()['message']['total-results']
    print(f'  Total journals: {total:,}')

    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            ckpt = json.load(f)
        rows = ckpt['rows']
        offset = ckpt['offset']
        print(f'  Resuming from offset {offset:,} ({len(rows):,} journals already downloaded)')

    with tqdm(total=total, initial=offset, desc='Crossref journals', unit='journal') as pbar:
        while offset < total:
            url = f'https://api.crossref.org/journals?rows={per_page}&offset={offset}&mailto={MAILTO}'
            for attempt in range(5):
                try:
                    r = session.get(url, timeout=30)
                    if r.status_code == 429:
                        time.sleep(10)
                        continue
                    r.raise_for_status()
                    break
                except Exception as e:
                    time.sleep(3)
            try:
                data = r.json()
                msg = data.get('message', {}) if isinstance(data, dict) else {}
                items = msg.get('items', []) if isinstance(msg, dict) else []
            except Exception:
                items = []
            if not items:
                offset += per_page
                pbar.update(per_page)
                time.sleep(0.5)
                continue
            for j in items:
                issns = j.get('ISSN', [])
                rows.append({
                    'journal_name': j.get('title', ''),
                    'issn_l': issns[0] if issns else None,
                    'issn': issns,
                    'publisher': j.get('publisher', ''),
                    'works_count': j.get('counts', {}).get('total-dois', 0),
                })
            offset += len(items)
            pbar.update(len(items))
            if offset % 5000 == 0:
                with open(CHECKPOINT, 'w') as f:
                    json.dump({'rows': rows, 'offset': offset}, f)
            time.sleep(0.05)

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    cr = pd.DataFrame(rows)
    cr = cr[cr['journal_name'].ne('')].reset_index(drop=True)
    cr.to_parquet(CROSSREF_CACHE, index=False)
    print(f'  Saved {len(cr):,} journals to cache')

# ── Step 2: Match proposal journal strings to Crossref ────────────────────────
print('\nLoading parsed proposal journal strings...')
parsed = pd.read_parquet(WORK_DIR / 'intermediate/proposal_atypicality_ra_handoff/proposal_references_parsed.parquet')
strings = parsed['parsed_journal_raw'].dropna()

def looks_like_journal(s):
    s = str(s).strip()
    if len(s) < 3 or len(s) > 100: return False
    if re.search(r'\d{4}', s): return False
    if len(re.findall(r'[a-zA-Z]', s)) < 3: return False
    if re.search(r'https?://', s): return False
    return True

counts = strings[strings.map(looks_like_journal)].value_counts()
targets = counts[counts >= 3].index.tolist()
print(f'Proposal journal strings to match: {len(targets):,}')

STOPWORDS = {'of', 'the', 'and', 'for', 'in', 'on', 'to', 'a', 'an', '&'}
ABBREV = {
    'journal': 'j', 'journals': 'j', 'proceedings': 'proc', 'national': 'natl',
    'academy': 'acad', 'sciences': 'sci', 'science': 'sci', 'nature': 'nat',
    'scientific': 'sci', 'american': 'am', 'british': 'br', 'european': 'eur',
    'international': 'int', 'clinical': 'clin', 'medicine': 'med', 'medical': 'med',
    'biology': 'biol', 'biological': 'biol', 'biochemistry': 'biochem', 'chemistry': 'chem',
    'molecular': 'mol', 'genetics': 'genet', 'immunology': 'immunol', 'neuroscience': 'neurosci',
    'physiology': 'physiol', 'pharmacology': 'pharmacol', 'review': 'rev', 'reviews': 'rev',
    'letters': 'lett', 'research': 'res', 'reports': 'rep', 'communications': 'commun',
    'current': 'curr', 'development': 'dev', 'experimental': 'exp', 'therapy': 'ther',
    'epidemiology': 'epidemiol', 'nutrition': 'nutr', 'technology': 'technol',
    'biotechnology': 'biotechnol', 'environmental': 'environ', 'cardiovascular': 'cardiovasc',
    'oncology': 'oncol', 'neurology': 'neurol', 'endocrinology': 'endocrinol',
}

def norm(text):
    text = str(text).lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(t for t in text.split() if t not in STOPWORDS)

def abbrev(text):
    toks = norm(text).split()
    return ' '.join(ABBREV.get(t, t[:4] if len(t) > 6 else t) for t in toks)

# Build lookup index from Crossref
print('Building Crossref lookup index...')
cr['norm'] = cr['journal_name'].map(norm)
cr['abbr'] = cr['journal_name'].map(abbrev)
norm_to_idx = {}
abbr_to_idx = {}
for i, row in cr.iterrows():
    if row['norm']: norm_to_idx.setdefault(row['norm'], i)
    if row['abbr']: abbr_to_idx.setdefault(row['abbr'], i)

print('Matching proposal strings to Crossref...')
matched_issns = set()
match_results = {}

for query in tqdm(targets, desc='Matching to Crossref'):
    q_norm = norm(query)
    q_abbr = abbrev(query)

    idx = norm_to_idx.get(q_norm) or abbr_to_idx.get(q_abbr) or norm_to_idx.get(q_abbr)
    if idx is not None:
        row = cr.iloc[idx]
        if row['issn_l']:
            matched_issns.add(row['issn_l'])
            match_results[query] = row['issn_l']

print(f'Matched {len(match_results):,} / {len(targets):,} strings → {len(matched_issns):,} unique ISSNs')

# ── Step 3: Use ISSNs as source IDs (OpenAlex rate limited) ──────────────────
# We use ISSN-L as the source ID. Moh can map these to OpenAlex IDs via
# SciSciNet's ISSN mappings on his end.
print(f'\nUsing ISSN-L as source IDs (OpenAlex rate limited today).')
print(f'Moh can join these to OpenAlex IDs via ISSN on his end.')

# ── Build final authority parquet ─────────────────────────────────────────────
print('\nBuilding authority parquet...')
rows = []
for _, row in cr.iterrows():
    issns = row['issn'] if isinstance(row['issn'], list) else []
    issn_l = row['issn_l']
    rows.append({
        'journal_name': row['journal_name'],
        'openalex_source_id': issn_l,  # using ISSN-L as proxy source ID
        'issn_l': issn_l,
        'issn': issns,
        'abbreviated_title': None,
        'alternate_titles': [],
        'publisher': row.get('publisher'),
        'homepage_url': None,
        'works_count': row.get('works_count'),
        'cited_by_count': None,
        'oa_2yr_mean_citedness': None,
        'oa_h_index': None,
        'oa_i10_index': None,
        'is_oa': None,
        'is_in_doaj': None,
        'is_core': None,
        'host_organization_name': None,
        'mag_id': None,
        'wikidata_id': None,
        'fatcat_id': None,
    })

df = pd.DataFrame(rows)
df.to_parquet(OUT_PATH, index=False)
scoreable = df['openalex_source_id'].notna().sum()
print(f'\nSaved {len(df):,} journals to {OUT_PATH}')
print(f'With OpenAlex source ID (scoreable): {scoreable:,}')
print(f'\nDone! Now run: python3 run_pipeline.py')
