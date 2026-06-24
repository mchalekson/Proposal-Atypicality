"""
Targeted OpenAlex journal lookup.
Instead of downloading all 229k journals, looks up only the journal strings
that actually appear in the proposal reference list (5+ times).
Results saved to Data/openalex_sources_journals.parquet.
"""
import requests, pandas as pd, re, time, json
from pathlib import Path
from tqdm.auto import tqdm

WORK_DIR = Path(__file__).parent
DATA_DIR = WORK_DIR.parent / 'Data'
OUT_PATH = DATA_DIR / 'openalex_sources_journals.parquet'
CACHE_PATH = DATA_DIR / 'openalex_lookup_cache.json'
MAILTO = 'mchalekson@gmail.com'

# ── Get unique journal strings from parsed refs ────────────────────────────────
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
# Focus on strings appearing 3+ times — covers the vast majority of citation volume
targets = counts[counts >= 3].index.tolist()
print(f'Unique journal strings to look up: {len(targets):,}')

# ── Load cache ─────────────────────────────────────────────────────────────────
cache = {}
if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    print(f'Loaded {len(cache):,} cached lookups')

remaining = [t for t in targets if t not in cache]
print(f'Remaining to look up: {len(remaining):,}')

# ── Lookup function ────────────────────────────────────────────────────────────
def search_openalex_source(query):
    """Search OpenAlex for a journal by name. Returns best match dict or None."""
    url = f'https://api.openalex.org/sources?search={requests.utils.quote(query)}&filter=type:journal&per_page=3&mailto={MAILTO}'
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get('Retry-After', 15))
                print(f'  Rate limited, waiting {wait}s...')
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return None
            results = r.json().get('results', [])
            if not results:
                return None
            # Return the best match (first result)
            s = results[0]
            stats = s.get('summary_stats', {}) or {}
            return {
                'journal_name': s.get('display_name'),
                'openalex_source_id': s.get('id', '').replace('https://openalex.org/', ''),
                'issn_l': s.get('issn_l'),
                'issn': s.get('issn') or [],
                'abbreviated_title': None,
                'alternate_titles': s.get('alternate_titles') or [],
                'publisher': s.get('host_organization_name'),
                'homepage_url': s.get('homepage_url'),
                'works_count': s.get('works_count'),
                'cited_by_count': s.get('cited_by_count'),
                'oa_2yr_mean_citedness': stats.get('2yr_mean_citedness'),
                'oa_h_index': stats.get('h_index'),
                'oa_i10_index': stats.get('i10_index'),
                'is_oa': s.get('is_oa'),
                'is_in_doaj': s.get('is_in_doaj'),
                'is_core': s.get('is_core'),
                'host_organization_name': s.get('host_organization_name'),
                'mag_id': (s.get('ids') or {}).get('mag'),
                'wikidata_id': (s.get('ids') or {}).get('wikidata'),
                'fatcat_id': None,
            }
        except Exception as e:
            print(f'  Error: {e}')
            time.sleep(3)
    return None

# ── Run lookups ────────────────────────────────────────────────────────────────
with tqdm(total=len(remaining), desc='Looking up journals', unit='journal') as pbar:
    for i, query in enumerate(remaining):
        result = search_openalex_source(query)
        cache[query] = result
        time.sleep(0.12)

        if (i + 1) % 200 == 0:
            with open(CACHE_PATH, 'w') as f:
                json.dump(cache, f)
            found = sum(1 for v in cache.values() if v is not None)
            pbar.set_postfix(matched=found, total_cached=len(cache))

        pbar.update(1)

# Final cache save
with open(CACHE_PATH, 'w') as f:
    json.dump(cache, f)

# ── Build deduplicated authority parquet from cache ───────────────────────────
rows = [v for v in cache.values() if v is not None]
df = pd.DataFrame(rows).drop_duplicates('openalex_source_id')
df.to_parquet(OUT_PATH, index=False)
print(f'\nSaved {len(df):,} unique journals to {OUT_PATH}')
print(f'Lookup hit rate: {len(df):,} / {len(targets):,} strings ({100*len(df)/len(targets):.0f}%)')
