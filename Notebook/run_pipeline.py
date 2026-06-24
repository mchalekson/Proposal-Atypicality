"""
Run the proposal atypicality pipeline (Steps 1-8) as a standalone script.
Matches the notebook logic exactly.
"""
from pathlib import Path
import json, os, re, shutil, subprocess, time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR = Path(__file__).parent
DATA_DIR = WORK_DIR.parent / 'Data'
INPUT_PARQUET = DATA_DIR / 'research_project_texts_2006-2026.parquet'

OPENALEX_SOURCES_PATH   = DATA_DIR / 'openalex_sources_journals.parquet'
SCISCINET_SOURCES_PATH  = DATA_DIR / 'sciscinet_sources.parquet'
NNF_JOURNALS_PATH       = DATA_DIR / 'List of Journals of Papers Produced by NNF Proposals - 2026-06-16.parquet'
WOS_JCR_PATH            = DATA_DIR / 'wos_jcr.csv'
MANUAL_JOURNAL_MATCH_CSV_PATH = DATA_DIR / 'manual_journal_matches.csv'

SCISCINET_PAPERREFS_PATH    = DATA_DIR / 'sciscinet_paperrefs.parquet'
SCISCINET_PAPERS_PATH       = DATA_DIR / 'sciscinet_papers.parquet'
SCISCINET_PAPERSOURCES_PATH = DATA_DIR / 'sciscinet_papersources.parquet'

ID_COL   = 'Application id'
YEAR_COL = 'app year'
REF_COL  = 'Project literature references clean'

INTERMEDIATE_DIR = WORK_DIR / 'intermediate' / 'proposal_atypicality_ra_handoff'
OUTPUT_DIR       = WORK_DIR / 'output'       / 'proposal_atypicality_ra_handoff'
CSV_DIR          = INTERMEDIATE_DIR / 'csv'
for p in [INTERMEDIATE_DIR, OUTPUT_DIR, CSV_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RAW_REFS_PATH                       = INTERMEDIATE_DIR / 'proposal_references_raw.parquet'
PARSED_REFS_PATH                    = INTERMEDIATE_DIR / 'proposal_references_parsed.parquet'
PARSING_FAILURES_PATH               = INTERMEDIATE_DIR / 'proposal_references_parsing_failures.parquet'
PARSER_STATUS_PATH                  = OUTPUT_DIR / 'reference_parser_status.csv'
PARSER_USAGE_PATH                   = OUTPUT_DIR / 'reference_parser_usage_summary.csv'
JOURNAL_AUTHORITY_PATH              = INTERMEDIATE_DIR / 'journal_authority.parquet'
JOURNAL_AUTHORITY_ALIASES_PATH      = INTERMEDIATE_DIR / 'journal_authority_aliases.parquet'
REFERENCE_JOURNAL_MATCHES_PATH      = INTERMEDIATE_DIR / 'proposal_reference_journal_matches.parquet'
UNMATCHED_REFERENCE_JOURNAL_STRINGS_PATH = INTERMEDIATE_DIR / 'unmatched_reference_journal_strings.parquet'
PROPOSAL_MATCH_SUMMARY_PATH         = OUTPUT_DIR / 'proposal_reference_journal_match_summary.parquet'
PROPOSAL_JOURNAL_PAIRS_PATH         = INTERMEDIATE_DIR / 'proposal_openalex_source_pairs.parquet'

DEDUP_MATCHED_JOURNALS_PATH   = OUTPUT_DIR / 'deduplicated_reference_journal_matches.parquet'
DEDUP_UNMATCHED_JOURNALS_PATH = OUTPUT_DIR / 'deduplicated_unmatched_reference_journals.parquet'

print('INPUT_PARQUET exists:', INPUT_PARQUET.exists())

# ── Step 1: Parser activation ──────────────────────────────────────────────────
ANYSTYLE_BIN = '/opt/homebrew/lib/ruby/gems/4.0.0/bin/anystyle'
USE_ANYSTYLE = True
REQUIRE_ANYSTYLE = False   # regex fallback allowed
ALLOW_REGEX_FALLBACK = True
STOP_IF_REQUIRED_PARSER_INACTIVE = False

anystyle_active = Path(ANYSTYLE_BIN).exists() or shutil.which(ANYSTYLE_BIN) is not None
print(f'anystyle active: {anystyle_active} ({ANYSTYLE_BIN})')

parser_status_df = pd.DataFrame({
    'anystyle': {'requested': True, 'required': REQUIRE_ANYSTYLE, 'active': anystyle_active, 'message': ANYSTYLE_BIN}
}).T
parser_status_df.to_csv(PARSER_STATUS_PATH)

# ── Utility functions ──────────────────────────────────────────────────────────
DOI_RE  = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.I)
PMID_RE = re.compile(r'\bPMID\s*:?\s*(\d+)\b', re.I)
PMCID_RE = re.compile(r'\b(PMC\d+)\b', re.I)
YEAR_RE  = re.compile(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)')
MONTH_RE = re.compile(r'^(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)$', re.I)

def normalize_space(text):
    text = str(text).replace('\r', '\n').replace(' ', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_doi(doi):
    doi = str(doi).strip()
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi, flags=re.I)
    doi = re.sub(r'^doi\s*:\s*', '', doi, flags=re.I)
    return doi.strip().rstrip('.,;:)\]}>').lower()

def split_references(text):
    text = normalize_space(text)
    if not text:
        return []
    text = re.sub(r'^(references|bibliography)\s*[:\n]+', '', text, flags=re.I).strip()
    numbered = re.split(r'(?m)(?=^\s*(?:\[\d{1,3}\]|\(\d{1,3}\)|\*?\d{1,3}[\.)])\s+)', text)
    numbered = [x.strip() for x in numbered if len(x.strip()) > 20]
    if len(numbered) >= 2:
        return numbered
    blankline = re.split(r'\n\s*\n+', text)
    blankline = [x.strip() for x in blankline if len(x.strip()) > 20]
    if len(blankline) >= 2:
        return blankline
    newline = re.split(r'\n+', text)
    newline = [x.strip() for x in newline if len(x.strip()) > 40]
    if len(newline) >= 2:
        return newline
    return [text]

# ── Step 2: Load and split references ─────────────────────────────────────────
print('\n=== Step 2: Load proposals ===')
df = pd.read_parquet(INPUT_PARQUET, engine='pyarrow')
refs = df[[ID_COL, YEAR_COL, REF_COL]].copy()
refs[REF_COL] = refs[REF_COL].fillna('').astype(str)
refs_nonempty = refs[refs[REF_COL].str.strip().ne('')].copy()
print(f'proposal rows: {len(df)}, non-empty ref rows: {len(refs_nonempty)}')

raw_rows = []
for _, row in tqdm(refs_nonempty.iterrows(), total=len(refs_nonempty), desc='Splitting refs'):
    for pos, entry in enumerate(split_references(row[REF_COL]), start=1):
        raw_rows.append({ID_COL: row[ID_COL], YEAR_COL: row[YEAR_COL], 'reference_position': pos, 'reference_text': entry})
raw_refs = pd.DataFrame(raw_rows)
raw_refs.to_parquet(RAW_REFS_PATH, index=False)
print(f'raw reference entries: {len(raw_refs)}')

# ── Step 3: Parse references ───────────────────────────────────────────────────
print('\n=== Step 3: Parse references ===')

def has_value(x):
    return pd.notna(x) and str(x).strip() != ''

def classify_reference(ref_text, parsed_journal_raw=None, parsed_doi=None):
    ref_text = '' if pd.isna(ref_text) else str(ref_text)
    labels = []
    lower = ref_text.lower()
    if re.search(r'\b(book|chapter|edited by|publisher|press|isbn)\b', lower): labels.append('book_signal')
    if re.search(r'\b(thesis|dissertation)\b', lower): labels.append('thesis_signal')
    if re.search(r'\b(report|working paper|white paper|technical report)\b', lower): labels.append('report_signal')
    if re.search(r'\b(conference|proceedings|symposium|workshop)\b', lower): labels.append('conference_signal')
    has_paper_signal = bool(re.search(r'\b(journal|vol\.|volume|issue|pages?|pp\.|doi)\b', lower))
    has_pages_or_volume = bool(re.search(r'\b\d{1,4}\s*\(?\d{0,4}\)?\s*[:,]\s*\d{1,6}', ref_text))
    has_year = bool(YEAR_RE.search(ref_text))
    has_doi = has_value(parsed_doi) or bool(DOI_RE.search(ref_text))
    has_journal = has_value(parsed_journal_raw)
    if has_journal or has_doi or has_paper_signal or has_pages_or_volume:
        ref_type = 'likely_paper'
    elif labels:
        ref_type = 'likely_nonpaper'
    elif has_year:
        ref_type = 'ambiguous'
    else:
        ref_type = 'unclassified'
    return ref_type, ';'.join(labels)

def anystyle_item_to_parsed(item):
    journal = item.get('container-title') or item.get('journal') or item.get('collection-title')
    title = item.get('title')
    if isinstance(title, list): title = ' '.join(map(str, title))
    if isinstance(journal, list): journal = ' '.join(map(str, journal))
    return {
        'parsed_title': title if title else pd.NA,
        'parsed_journal_raw': journal if journal else pd.NA,
        'parsed_year': item.get('date') or item.get('year') or pd.NA,
        'parsed_doi': normalize_doi(item.get('doi')) if item.get('doi') else pd.NA,
        'parsed_volume': item.get('volume') or pd.NA,
        'parsed_issue': item.get('issue') or pd.NA,
        'parsed_pages': item.get('pages') or item.get('page') or pd.NA,
    }

def batch_anystyle(reference_texts, tmp_dir):
    """Parse all references in one AnyStyle call. Returns list of parsed dicts."""
    import tempfile
    # AnyStyle requires one reference per line; newlines within a ref must be stripped
    cleaned = [re.sub(r'\s+', ' ', t).strip() for t in reference_texts]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', dir=tmp_dir, delete=False, encoding='utf-8') as f:
        f.write('\n'.join(cleaned))
        tmp_in = f.name
    tmp_out = tmp_in.replace('.txt', '.json')
    try:
        proc = subprocess.run(
            [ANYSTYLE_BIN, '--stdout', '-f', 'json', 'parse', tmp_in],
            capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return [{}] * len(reference_texts)
        results = json.loads(proc.stdout)
        if not isinstance(results, list):
            return [{}] * len(reference_texts)
        # Pad/truncate to match input length
        while len(results) < len(reference_texts):
            results.append({})
        return results[:len(reference_texts)]
    except Exception:
        return [{}] * len(reference_texts)
    finally:
        for p in [tmp_in, tmp_out]:
            try: os.unlink(p)
            except: pass

def clean_candidate_span(span):
    span = str(span)
    span = re.sub(r'^\s*(in\s+)?', '', span, flags=re.I)
    span = re.sub(r'\b(doi|pmid|pmcid)\b.*$', '', span, flags=re.I)
    span = re.sub(r'\b(?:19\d{2}|20\d{2})\b.*$', '', span)
    span = re.sub(r'\b\d{1,4}\s*\(?\d{0,4}\)?\s*[:,].*$', '', span)
    span = re.sub(r'[,.;:()\[\]"""]+$', '', span).strip()
    span = re.sub(r'^[,.;:()\[\]"""]+', '', span).strip()
    return re.sub(r'\s+', ' ', span)

def regex_journal_candidates(ref_text):
    text = normalize_space(ref_text).replace('\n', ' ')
    text = re.sub(r'^\s*(?:\[\d+\]|\(\d+\)|\*?\d+[\.)])\s*', '', text)
    candidates = []
    patterns = [
        r'\.\s*([^\.]{2,90}?)\.\s*(?:19\d{2}|20\d{2})\b',
        r'\.\s*([^\.]{2,90}?),\s*(?:19\d{2}|20\d{2})\b',
        r'\.\s*([^\.]{2,70}?)\s+\d{1,4}\s*[,;:]\s*\d+',
        r'\b([A-Z][A-Za-z&\. ]{2,70}?)\s+\d{1,4}\s*,\s*\d+',
        r'\.\s*([^\.]{2,90}?),\s*\d{1,4}\s*\(',
        r'\.\s*([^\.]{2,90}?),\s*\d{1,4}\s*[:,]',
    ]
    for pat in patterns:
        for match in re.finditer(pat, text):
            cand = clean_candidate_span(match.group(1))
            if 2 <= len(cand) <= 100:
                candidates.append(cand)
    cleaned = []
    bad = re.compile(r'\b(et al|abstract|appendix|table|figure|references|retrieved|available|http|doi)\b', re.I)
    for cand in candidates:
        toks = cand.split()
        if bad.search(cand) or MONTH_RE.match(cand):
            continue
        if len(toks) <= 12 and any(re.search('[A-Za-z]', t) for t in toks):
            cleaned.append(cand)
    return list(dict.fromkeys(cleaned))

def parse_reference(reference_text):
    errors = []
    parsed = {
        'parsed_title': pd.NA, 'parsed_journal_raw': pd.NA, 'parsed_year': pd.NA,
        'parsed_doi': pd.NA, 'parsed_volume': pd.NA, 'parsed_issue': pd.NA,
        'parsed_pages': pd.NA, 'parser_used': 'none', 'parse_success': False, 'parse_error': '',
    }
    if USE_ANYSTYLE and anystyle_active:
        try:
            aparsed, err = parse_anystyle_citation(reference_text)
            if err: errors.append('anystyle:' + err)
            parsed.update({k: v for k, v in aparsed.items() if pd.notna(v)})
            if pd.notna(parsed.get('parsed_journal_raw')):
                parsed['parser_used'] = 'anystyle'
                parsed['parse_success'] = True
                parsed['parse_error'] = ' | '.join(errors)
                return parsed
        except Exception as exc:
            errors.append('anystyle:' + repr(exc))
    if ALLOW_REGEX_FALLBACK:
        candidates = regex_journal_candidates(reference_text)
        if candidates:
            parsed['parsed_journal_raw'] = candidates[0]
            parsed['parser_used'] = 'regex_fallback'
            parsed['parse_success'] = True
            parsed['parse_error'] = ' | '.join(errors)
            return parsed
    parsed['parse_error'] = ' | '.join(errors)
    return parsed

# Batch anystyle for speed: parse all at once using --finder stdin
# AnyStyle processes one ref per invocation when called naively; to be faster we'll
# call it per-row but tqdm will show progress. For 100k refs this might take a while.
# We cache to PARSED_REFS_PATH so re-runs are instant.
if PARSED_REFS_PATH.exists():
    print(f'Loading cached parsed refs from {PARSED_REFS_PATH}')
    parsed_refs = pd.read_parquet(PARSED_REFS_PATH)
else:
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    BATCH_SIZE = 5000
    texts = raw_refs['reference_text'].tolist()
    all_anystyle = []
    if USE_ANYSTYLE and anystyle_active:
        print(f'Running AnyStyle in batches of {BATCH_SIZE} on {len(texts):,} references...')
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc='AnyStyle batches'):
            batch = texts[i:i+BATCH_SIZE]
            all_anystyle.extend(batch_anystyle(batch, tmp_dir))
        print(f'AnyStyle done. Got {len(all_anystyle)} results.')
    else:
        all_anystyle = [{}] * len(texts)

    parsed_rows = []
    for idx, row in enumerate(raw_refs.itertuples(index=False)):
        item = all_anystyle[idx] if idx < len(all_anystyle) else {}
        if item:
            parsed = anystyle_item_to_parsed(item)
            parsed['parser_used'] = 'anystyle' if pd.notna(parsed.get('parsed_journal_raw')) else 'anystyle_no_journal'
            parsed['parse_success'] = pd.notna(parsed.get('parsed_journal_raw'))
            parsed['parse_error'] = ''
        else:
            parsed = {'parsed_title': pd.NA, 'parsed_journal_raw': pd.NA, 'parsed_year': pd.NA,
                      'parsed_doi': pd.NA, 'parsed_volume': pd.NA, 'parsed_issue': pd.NA,
                      'parsed_pages': pd.NA, 'parser_used': 'none', 'parse_success': False, 'parse_error': ''}

        # Regex fallback if anystyle didn't get a journal
        if not pd.notna(parsed.get('parsed_journal_raw')) and ALLOW_REGEX_FALLBACK:
            candidates = regex_journal_candidates(row.reference_text)
            if candidates:
                parsed['parsed_journal_raw'] = candidates[0]
                parsed['parser_used'] = 'regex_fallback'
                parsed['parse_success'] = True

        dois  = sorted(set(normalize_doi(x) for x in DOI_RE.findall(row.reference_text)))
        dois  = [x for x in dois if x.startswith('10.')]
        pmids  = sorted(set(PMID_RE.findall(row.reference_text)))
        pmcids = sorted(set(x.upper() for x in PMCID_RE.findall(row.reference_text)))
        years  = YEAR_RE.findall(row.reference_text)
        ref_type, labels = classify_reference(row.reference_text, parsed.get('parsed_journal_raw'), parsed.get('parsed_doi'))
        parsed_rows.append({
            ID_COL: getattr(row, '_0'), YEAR_COL: getattr(row, '_1'),
            'reference_position': row.reference_position,
            'reference_text': row.reference_text,
            'dois': dois, 'pmids': pmids, 'pmcids': pmcids,
            'reference_year_guess': int(years[-1]) if years else pd.NA,
            'reference_type': ref_type, 'nonpaper_labels': labels,
            'is_paper_candidate': ref_type in ['likely_paper', 'unknown_with_year'],
            'has_doi': bool(dois), **parsed,
        })

    parsed_refs = pd.DataFrame(parsed_rows)
    for col in ['parsed_volume', 'parsed_issue', 'parsed_pages', 'parsed_year']:
        if col in parsed_refs.columns:
            parsed_refs[col] = parsed_refs[col].astype(str).where(parsed_refs[col].notna(), pd.NA)
    parsed_refs.to_parquet(PARSED_REFS_PATH, index=False)
    failures = parsed_refs[~parsed_refs['parse_success'] | parsed_refs['parsed_journal_raw'].isna()].copy()
    failures.to_parquet(PARSING_FAILURES_PATH, index=False)
    parser_usage = parsed_refs.groupby('parser_used', dropna=False).agg(
        references=('reference_text', 'size'),
        parse_success=('parse_success', 'sum'),
        parsed_journals=('parsed_journal_raw', lambda x: x.notna().sum()),
    ).reset_index()
    parser_usage.to_csv(PARSER_USAGE_PATH, index=False)
    print(parser_usage.to_string())

print(f'parsed refs: {len(parsed_refs)}')
print(f'likely paper refs: {int(parsed_refs["is_paper_candidate"].sum())}')
print(f'refs with parsed journal: {int(parsed_refs["parsed_journal_raw"].notna().sum())}')

# ── Step 4: Build Journal Authority ───────────────────────────────────────────
print('\n=== Step 4: Build journal authority ===')

ABBREV = {
    'journal': 'j', 'journals': 'j', 'proceedings': 'proc', 'proceeding': 'proc',
    'national': 'natl', 'academy': 'acad', 'academies': 'acad',
    'sciences': 'sci', 'science': 'sci', 'nature': 'nat', 'nat.': 'nat', 'springer nature': 'nat',
    'scientific': 'sci', 'united': 'u', 'states': 's', 'america': 'a', 'american': 'am',
    'british': 'br', 'european': 'eur', 'international': 'int', 'clinical': 'clin',
    'clinic': 'clin', 'medicine': 'med', 'medical': 'med', 'biology': 'biol', 'biological': 'biol',
    'biochemistry': 'biochem', 'biochemical': 'biochem', 'chemistry': 'chem', 'chemical': 'chem',
    'molecular': 'mol', 'cellular': 'cell', 'genetics': 'genet', 'genetic': 'genet',
    'immunology': 'immunol', 'microbiology': 'microbiol', 'physiology': 'physiol',
    'pharmacology': 'pharmacol', 'endocrinology': 'endocrinol', 'metabolism': 'metab',
    'neuroscience': 'neurosci', 'neurology': 'neurol', 'cardiology': 'cardiol',
    'cardiovascular': 'cardiovasc', 'oncology': 'oncol', 'hematology': 'hematol',
    'haematology': 'haematol', 'physics': 'phys', 'physical': 'phys', 'review': 'rev',
    'reviews': 'rev', 'letters': 'lett', 'materials': 'mater', 'applied': 'appl',
    'environmental': 'environ', 'technology': 'technol', 'biotechnology': 'biotechnol',
    'research': 'res', 'reports': 'rep', 'communications': 'commun', 'current': 'curr',
    'opinion': 'opin', 'development': 'dev', 'experimental': 'exp', 'therapy': 'ther',
    'translational': 'transl', 'epidemiology': 'epidemiol', 'nutrition': 'nutr', 'respiratory': 'respir',
    'computational': 'comput', 'systems': 'syst', 'applications': 'appl', 'engineering': 'eng', 'engineer': 'eng',
}
STOPWORDS = {'of', 'the', 'and', 'for', 'in', 'on', 'to', 'a', 'an', '&'}
KNOWN_ALIASES = {
    # PNAS variants — authority has "Proceedings of the National Academy of Sciences"
    'nejm': 'new england journal medicine',
    'new engl j med': 'new england journal medicine',
    'n engl j med': 'new england journal medicine',
    'pnas': 'proceedings national academy sciences',
    'proc natl acad sci u s a': 'proceedings national academy sciences',
    'proc natl acad sci usa': 'proceedings national academy sciences',
    'natl acad sci u s a': 'proceedings national academy sciences',
    'proc natl acad sci': 'proceedings national academy sciences',
    'proc. natl. acad. sci. u. s. a': 'proceedings national academy sciences',
    'natl. acad. sci. u. s. a': 'proceedings national academy sciences',
    'proceedings national academy sciences united states america': 'proceedings national academy sciences',
    # Annals
    'ann intern med': 'annals internal medicine',
    'ann rheum dis': 'annals rheumatic diseases',
    'ann surg': 'annals surgery',
    'ann neurol': 'annals neurology',
    # Infectious disease
    'j virol': 'journal virology',
    'clin infect dis': 'clinical infectious diseases',
    'j infect dis': 'journal infectious diseases',
    # Cardiology/pulmonary
    'chest': 'chest',
    # Botany
    'new phytol': 'new phytologist',
    'front plant sci': 'frontiers plant science',
    # Internal medicine
    'j intern med': 'journal internal medicine',
    # Oncology
    'acta oncol': 'acta oncologica',
    # Chemistry
    'chem. soc. rev': 'chemical society reviews',
    'chem soc rev': 'chemical society reviews',
    'acs catal': 'acs catalysis',
    'j med chem': 'journal medicinal chemistry',
    'metab eng': 'metabolic engineering',
    # Fertility
    'fertil steril': 'fertility sterility',
    # Pathology
    'am j pathol': 'american journal pathology',
    # Microbiology
    'microb cell fact': 'microbial cell factories',
    # Biochemistry
    'biochim biophys acta': 'biochimica biophysica acta',
    'free radic biol med': 'free radical biology medicine',
    # Physiology (truncated variants)
    'am j physiol endocrinol metab': 'american journal physiology endocrinology metabolism',
    'am j physiol renal physiol': 'american journal physiology renal physiology',
    'am j physiol cell physiol': 'american journal physiology cell physiology',
    'am j physiol heart circ physiol': 'american journal physiology heart circulatory physiology',
    # Frontiers (with location suffix stripped)
    'front endocrinol lausanne': 'frontiers endocrinology',
    'front endocrinol (lausanne': 'frontiers endocrinology',
    # Obstetrics/gynecology
    'am j obstet gynecol': 'american journal obstetrics gynecology',
    # Annual reviews
    'annu rev biochem': 'annual review biochemistry',
    # Botany
    'j exp bot': 'journal experimental botany',
    # Obesity (truncated)
    'int j obes (lond': 'international journal obesity',
    'int j obes': 'international journal obesity',
    # Optics
    'opt. express': 'optics express',
    # Computational chemistry
    'j. chem. theory comput': 'journal chemical theory computation',
    'j chem theory comput': 'journal chemical theory computation',
    # Nursing
    'j adv nurs': 'journal advanced nursing',
    # Sports medicine
    'med sci sports exerc': 'medicine science sports exercise',
    # Infectious disease
    'infect immun': 'infection immunity',
    'plos pathog': 'plos pathogens',
    # Internal medicine
    'arch intern med': 'archives internal medicine',
    # Neurology
    'neurobiol dis': 'neurobiology disease',
    # Nature reviews disease primers
    'nat rev dis primers': 'nature reviews disease primers',
    # Synthetic biology
    'acs synth biol': 'acs synthetic biology',
    # Danish medical journal
    'dan med j': 'danish medical journal',
    # Video methods
    'j vis exp': 'journal visualized experiments',
    # Angew chem
    'chem. int. ed': 'angewandte chemie international edition',
    'engl. j. med': 'new england journal medicine',
    'natl. acad. sci': 'proceedings national academy sciences',
    # JCI
    'jci': 'journal clinical investigation',
    'j clin invest': 'journal clinical investigation',
    # Chemistry
    'j. am. chem. soc': 'journal american chemical society',
    'j am chem soc': 'journal american chemical society',
    'jacs': 'journal american chemical society',
    'angew chem int ed': 'angewandte chemie international edition',
    'chem. int. ed': 'angewandte chemie international edition',
    'angew. chem. int. ed': 'angewandte chemie international edition',
    # Frontiers journals
    'front immunol': 'frontiers immunology',
    'front microbiol': 'frontiers microbiology',
    'front physiol': 'frontiers physiology',
    'front neurosci': 'frontiers neuroscience',
    'front oncol': 'frontiers oncology',
    'front genet': 'frontiers genetics',
    'front cell dev biol': 'frontiers cell developmental biology',
    'front endocrinol': 'frontiers endocrinology',
    # Scandinavian/public health
    'scand j public health': 'scandinavian journal public health',
    # Biochemistry
    'biochim biophys acta': 'biochimica biophysica acta',
    'bba': 'biochimica biophysica acta',
    # Bone/nephrology
    'j bone miner res': 'journal bone mineral research',
    'j am soc nephrol': 'journal american society nephrology',
    'jasn': 'journal american society nephrology',
    # Nature reviews
    'nat rev drug discov': 'nature reviews drug discovery',
    'nat rev cancer': 'nature reviews cancer',
    'nat rev immunol': 'nature reviews immunology',
    'nat rev genet': 'nature reviews genetics',
    'nat rev mol cell biol': 'nature reviews molecular cell biology',
    'nat rev neurosci': 'nature reviews neuroscience',
    'nat rev cardiol': 'nature reviews cardiology',
    # Reproduction/endocrinology
    'hum reprod': 'human reproduction',
    'am j physiol endocrinol metab': 'american journal physiology endocrinology metabolism',
    'diabet med': 'diabetic medicine',
    'endocr rev': 'endocrine reviews',
    # Genetics/molecular
    'hum mol genet': 'human molecular genetics',
    'am j hum genet': 'american journal human genetics',
    'nat struct mol biol': 'nature structural molecular biology',
    # Vascular
    'arterioscler thromb vasc biol': 'arteriosclerosis thrombosis vascular biology',
    'j cereb blood flow metab': 'journal cerebral blood flow metabolism',
    # Cardiology
    'j am heart assoc': 'journal american heart association',
    'eur heart j': 'european heart journal',
    # Development
    'dev cell': 'developmental cell',
    # Other
    'bmj': 'british medical journal',
    'am. j. hum. genet': 'american journal human genetics',
    'acta neurol scand': 'acta neurologica scandinavica',
    'glob med genet': 'global medical genetics',
    'j gen intern med': 'journal general internal medicine',
    'nanoen': 'nano energy',
    'sens diagn': 'sensors diagnostics',
    'molcel': 'mol cell',
    'celrep': 'cell reports',
    'j bone jt infect': 'journal bone joint infection',
    'jbji': 'journal bone joint infection',
    'lipids health dis': 'lipids health disease',
    'nat protoc': 'nature protocols',
    'cmet': 'cell metabolism',
    'jnucmat': 'journal nuclear materials',
    'omtn': 'molecular therapy nucleic acids',
    'res in phys': 'results physics',
    'biotechnol bioeng': 'biotechnology bioengineering',
    'agrformet': 'agricultural forest meteorology',
    'sci adv': 'science advances',
}

METADATA_COLS = [
    'journal_name', 'authority_source', 'openalex_source_id', 'issn_l', 'issn', 'abbreviated_title',
    'alternate_titles', 'publisher', 'homepage_url', 'works_count', 'cited_by_count', 'oa_2yr_mean_citedness',
    'oa_h_index', 'oa_i10_index', 'is_in_doaj', 'is_oa', 'is_core', 'host_organization_name',
    'mag_id', 'wikidata_id', 'fatcat_id'
]

def normalize_journal_text(text):
    text = str(text).lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    text = re.sub(r'\b(the)\b', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def remove_stopwords(norm):
    return ' '.join(t for t in str(norm).split() if t not in STOPWORDS)

def abbreviation_variant(norm):
    toks = [t for t in str(norm).split() if t not in STOPWORDS]
    return ' '.join(ABBREV.get(t, t[:4] if len(t) > 6 else t) for t in toks)

def initials_variant(norm):
    toks = [t for t in str(norm).split() if t not in STOPWORDS]
    return ''.join(t[0] for t in toks if t)

def compact_variant(norm):
    return str(norm).replace(' ', '')

def journal_variants(name):
    norm = normalize_journal_text(name)
    base = remove_stopwords(norm)
    variants = {norm, base, abbreviation_variant(norm), abbreviation_variant(base), initials_variant(norm), compact_variant(norm), compact_variant(base)}
    return {v for v in variants if v and len(v) >= 2}

def ensure_cols(frame, cols):
    frame = frame.copy()
    for col in cols:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[cols]

def parse_listish(value):
    if value is None or (isinstance(value, float) and pd.isna(value)): return []
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, list): return value
    value = str(value).strip()
    if value.lower() in {'', 'null', 'none', 'nan'}: return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list): return [str(x).strip() for x in parsed if str(x).strip()]
        return [str(parsed).strip()]
    except Exception:
        return [value]

def read_openalex_sources(path):
    if not path.exists(): return pd.DataFrame(columns=METADATA_COLS)
    out = pd.read_parquet(path)
    if 'source_type' in out.columns:
        out = out[out['source_type'].eq('journal')].copy()
    out['authority_source'] = 'openalex_sources'
    out = out[out['journal_name'].notna()].copy()
    return ensure_cols(out, METADATA_COLS)

def read_sciscinet_sources(path):
    if not path.exists(): return pd.DataFrame(columns=METADATA_COLS)
    src = pd.read_parquet(path).rename(columns={'sourceid': 'openalex_source_id', 'display_name': 'journal_name'})
    src['issn'] = src['issn'].map(parse_listish)
    src['issn_l'] = src['issn'].map(lambda xs: xs[0] if isinstance(xs, list) and len(xs) else pd.NA)
    src = src[src['journal_name'].notna()].copy()
    src['journal_name'] = src['journal_name'].astype(str).str.strip()
    src = src[src['journal_name'].ne('')]
    src = src[src['issn'].map(lambda xs: isinstance(xs, list) and len(xs) > 0)]
    non_journal_pat = re.compile(r'\b(ebooks?|conference|congress|symposium|proceedings|repository|workshop)\b', re.I)
    src = src[~src['journal_name'].str.contains(non_journal_pat, na=False)]
    src['authority_source'] = 'sciscinet_sources'
    return ensure_cols(src, METADATA_COLS)

def read_nnf_journals(path):
    if not path.exists(): return pd.DataFrame(columns=METADATA_COLS)
    out = pd.read_parquet(path)
    name_col = 'journal_name' if 'journal_name' in out.columns else out.columns[0]
    out = out[[name_col]].rename(columns={name_col: 'journal_name'}).dropna()
    out['authority_source'] = 'proposal_output_journal_list'
    return ensure_cols(out, METADATA_COLS)

def read_wos_jcr(path):
    if not path.exists(): return pd.DataFrame(columns=METADATA_COLS)
    raw = pd.read_csv(path, header=None)
    header_idx = raw.index[raw.iloc[:, 1].astype(str).str.lower().eq('full journal title')]
    if len(header_idx) == 0: raise ValueError('No Full Journal Title in wos_jcr.csv')
    out = raw.iloc[int(header_idx[0]) + 1:, [1]].rename(columns={1: 'journal_name'}).dropna()
    out['authority_source'] = 'wos_jcr'
    return ensure_cols(out, METADATA_COLS)

authority_parts = [
    read_openalex_sources(OPENALEX_SOURCES_PATH),
    read_sciscinet_sources(SCISCINET_SOURCES_PATH),
    read_nnf_journals(NNF_JOURNALS_PATH),
    read_wos_jcr(WOS_JCR_PATH),
]
authority = pd.concat(authority_parts, ignore_index=True)
authority['journal_name'] = authority['journal_name'].astype(str).str.strip()
authority = authority[authority['journal_name'].ne('')].copy()
authority['journal_norm'] = authority['journal_name'].map(normalize_journal_text)
authority['journal_key']  = authority['journal_norm'].map(remove_stopwords)
priority = {'openalex_sources': 0, 'sciscinet_sources': 1, 'proposal_output_journal_list': 2, 'wos_jcr': 3}
authority['source_priority'] = authority['authority_source'].map(priority).fillna(9)
authority = authority.sort_values(['journal_key', 'source_priority', 'journal_name']).drop_duplicates('journal_key').reset_index(drop=True)
authority['journal_id'] = ['J%06d' % i for i in range(1, len(authority) + 1)]
authority.to_parquet(JOURNAL_AUTHORITY_PATH, index=False)

alias_rows = []
def add_alias(journal_id, alias, alias_type, authority_source):
    if pd.isna(alias) or not str(alias).strip(): return
    alias = str(alias).strip()
    alias_norm = normalize_journal_text(alias)
    alias_rows.append({
        'journal_id': journal_id, 'alias': alias, 'alias_norm': alias_norm,
        'alias_key': remove_stopwords(alias_norm), 'alias_type': alias_type, 'authority_source': authority_source,
    })

for row in authority.itertuples(index=False):
    add_alias(row.journal_id, row.journal_name, 'canonical', row.authority_source)
    add_alias(row.journal_id, row.abbreviated_title, 'openalex_abbreviated_title', row.authority_source)
    for alt in parse_listish(row.alternate_titles):
        add_alias(row.journal_id, alt, 'openalex_alternate_title', row.authority_source)
    for variant in journal_variants(row.journal_name):
        add_alias(row.journal_id, variant, 'generated_variant', row.authority_source)

alias_cols = ['journal_id', 'alias', 'alias_norm', 'alias_key', 'alias_type', 'authority_source']
authority_aliases = pd.DataFrame(alias_rows, columns=alias_cols) if alias_rows else pd.DataFrame(columns=alias_cols)
authority_aliases = authority_aliases.dropna(subset=['alias_key']).drop_duplicates(['journal_id', 'alias_key'])
authority_aliases.to_parquet(JOURNAL_AUTHORITY_ALIASES_PATH, index=False)

variant_to_ids = defaultdict(list)
for row in authority_aliases.itertuples(index=False):
    for variant in {row.alias_norm, row.alias_key, abbreviation_variant(row.alias_norm), compact_variant(row.alias_norm), compact_variant(row.alias_key)}:
        if variant and len(variant) >= 2:
            variant_to_ids[variant].append(row.journal_id)

for alias, canonical_key in KNOWN_ALIASES.items():
    hits = authority[authority['journal_key'].eq(canonical_key)]
    if len(hits):
        variant_to_ids[alias].append(hits.iloc[0]['journal_id'])

journal_id_to_name = dict(zip(authority['journal_id'], authority['journal_name']))
journal_id_to_key  = dict(zip(authority['journal_id'], authority['journal_key']))
authority_by_first_token = defaultdict(list)
for row in authority.itertuples(index=False):
    toks = row.journal_key.split()
    if toks:
        authority_by_first_token[toks[0]].append(row)

print(f'authority journals: {len(authority)}')
print(authority['authority_source'].value_counts(dropna=False).to_string())
print(f'authority aliases: {len(authority_aliases)}')

journal_id_to_background_source_id = authority.set_index('journal_id')['openalex_source_id'].to_dict()
journal_id_to_authority_source = authority.set_index('journal_id')['authority_source'].to_dict()
print('with OpenAlex/SciSciNet source IDs:', int(authority['openalex_source_id'].notna().sum()))
print('without source IDs (helper only):', int(authority['openalex_source_id'].isna().sum()))

# ── Step 5: Match parsed journals ─────────────────────────────────────────────
print('\n=== Step 5: Match journals to authority ===')

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
    print('rapidfuzz available')
except Exception:
    HAS_RAPIDFUZZ = False
    print('rapidfuzz not available, using difflib')

JOURNAL_JUNK_TERMS = {
    '', 'na', 'n/a', 'none', 'vol', 'volume', 'issue', 'no', 'number', 'pp', 'page', 'pages',
    'chapter', 'book', 'press', 'publisher', 'isbn', 'abstract', 'poster', 'presentation',
    'january','february','march','april','may','june','july','august','september','october','november','december',
    'jan','feb','mar','apr','jun','jul','aug','sep','sept','oct','nov','dec',
}

def is_bad_journal_candidate(candidate):
    if pd.isna(candidate) or not str(candidate).strip(): return True
    raw = str(candidate).strip()
    norm = normalize_journal_text(raw)
    key = remove_stopwords(norm)
    if MONTH_RE.match(raw): return True
    if norm in JOURNAL_JUNK_TERMS or key in JOURNAL_JUNK_TERMS: return True
    if re.fullmatch(r'\d+', norm): return True
    if re.fullmatch(r'\d+\s*[-:]\s*\d+', norm): return True
    if re.match(r'^(vol|volume|issue|no|number|pp|pages?)\b', norm): return True
    alpha = re.sub(r'[^a-z]', '', norm)
    if len(alpha) < 3: return True
    return False

def choose_journal_id(ids):
    return sorted(set(ids), key=lambda x: len(journal_id_to_key.get(x, '')), reverse=True)[0]

def fuzzy_score(a, b):
    seq = SequenceMatcher(None, a, b).ratio()
    if HAS_RAPIDFUZZ:
        wratio = fuzz.WRatio(a, b) / 100
        token_set = fuzz.token_set_ratio(a, b) / 100
        return max(seq, wratio, token_set)
    return seq

def match_candidate_to_authority(candidate):
    if is_bad_journal_candidate(candidate): return None
    candidate = str(candidate).strip()
    norm = normalize_journal_text(candidate)
    key  = remove_stopwords(norm)
    variants = [norm, key, abbreviation_variant(norm), abbreviation_variant(key), compact_variant(norm), compact_variant(key)]
    variants = [v for v in dict.fromkeys(variants) if v and len(v) >= 2]

    for variant in variants:
        ids = variant_to_ids.get(variant, [])
        if ids:
            journal_id = choose_journal_id(ids)
            return {'match_method': 'candidate_exact_or_variant', 'journal_id': journal_id,
                    'journal_name': journal_id_to_name[journal_id], 'journal_raw': candidate, 'match_score': 1.0}

    toks = key.split()
    if len(key) >= 8 and toks:
        candidates_auth = authority_by_first_token.get(toks[0], [])
        scored = sorted([(fuzzy_score(key, r.journal_key), r) for r in candidates_auth], key=lambda x: x[0], reverse=True)
        if scored:
            best_score, best_row = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            score_gap = best_score - second_score
            if (best_score >= 0.93 and score_gap >= 0.04) or (best_score >= 0.98 and score_gap >= 0.01):
                # Reject fuzzy matches to very short journal names (likely garbage)
                if len(best_row.journal_key) < 5:
                    return {'match_method': 'unmatched_candidate', 'journal_id': pd.NA,
                            'journal_name': pd.NA, 'journal_raw': candidate, 'match_score': np.nan}
                return {'match_method': 'candidate_fuzzy_authority', 'journal_id': best_row.journal_id,
                        'journal_name': best_row.journal_name, 'journal_raw': candidate, 'match_score': float(best_score)}

    return {'match_method': 'unmatched_candidate', 'journal_id': pd.NA, 'journal_name': pd.NA,
            'journal_raw': candidate, 'match_score': np.nan}

if REFERENCE_JOURNAL_MATCHES_PATH.exists():
    print(f'Loading cached matches from {REFERENCE_JOURNAL_MATCHES_PATH}')
    reference_matches = pd.read_parquet(REFERENCE_JOURNAL_MATCHES_PATH)
else:
    match_rows = []
    for row in tqdm(parsed_refs.itertuples(index=False), total=len(parsed_refs), desc='Matching journals'):
        rec = (match_candidate_to_authority(row.parsed_journal_raw)
               if row.is_paper_candidate and pd.notna(row.parsed_journal_raw) else None)
        if rec is None:
            rec = {'match_method': 'not_attempted_or_no_candidate', 'journal_id': pd.NA,
                   'journal_name': pd.NA, 'journal_raw': pd.NA, 'match_score': np.nan}
        match_rows.append({
            ID_COL: getattr(row, '_0'), YEAR_COL: getattr(row, '_1'),
            'reference_position': row.reference_position, 'reference_type': row.reference_type,
            'is_paper_candidate': row.is_paper_candidate, 'has_doi': row.has_doi,
            'parser_used': row.parser_used, 'parse_success': row.parse_success,
            'parsed_journal_raw': row.parsed_journal_raw, **rec,
        })
    reference_matches = pd.DataFrame(match_rows)
    reference_matches['matched_authority_journal'] = reference_matches['journal_id'].notna()
    reference_matches['journal_background_source_id'] = reference_matches['journal_id'].map(journal_id_to_background_source_id)
    reference_matches['journal_authority_source'] = reference_matches['journal_id'].map(journal_id_to_authority_source)
    reference_matches['journal_final_id']   = reference_matches['journal_background_source_id']
    reference_matches['journal_final_name'] = reference_matches['journal_name']
    reference_matches['journal_match_tier'] = np.where(
        reference_matches['journal_background_source_id'].notna(), 'openalex_or_sciscinet_source',
        np.where(reference_matches['matched_authority_journal'], 'authority_without_source_id', 'none'))
    reference_matches['has_final_journal'] = reference_matches['journal_final_id'].notna()
    reference_matches.to_parquet(REFERENCE_JOURNAL_MATCHES_PATH, index=False)

print(reference_matches['match_method'].value_counts(dropna=False).to_string())
print('authority-matched refs:', int(reference_matches['matched_authority_journal'].sum()))
print('scoreable (OpenAlex/SciSciNet) refs:', int(reference_matches['has_final_journal'].sum()))

# ── Step 6: Manual corrections (if CSV exists) ────────────────────────────────
if MANUAL_JOURNAL_MATCH_CSV_PATH.exists():
    manual_df = pd.read_csv(MANUAL_JOURNAL_MATCH_CSV_PATH)
    manual_df = manual_df[['journal_raw', 'journal_id']].dropna().drop_duplicates()
    manual_map = manual_df.set_index('journal_raw')['journal_id']
    manual_name_map     = authority.set_index('journal_id')['journal_name'].to_dict()
    manual_source_map   = authority.set_index('journal_id')['openalex_source_id'].to_dict()
    manual_auth_src_map = authority.set_index('journal_id')['authority_source'].to_dict()
    manual_mask = reference_matches['is_paper_candidate'] & reference_matches['journal_raw'].isin(manual_map.index)
    reference_matches.loc[manual_mask, 'journal_id']   = reference_matches.loc[manual_mask, 'journal_raw'].map(manual_map)
    reference_matches.loc[manual_mask, 'journal_name'] = reference_matches.loc[manual_mask, 'journal_id'].map(manual_name_map)
    reference_matches.loc[manual_mask, 'journal_background_source_id'] = reference_matches.loc[manual_mask, 'journal_id'].map(manual_source_map)
    reference_matches.loc[manual_mask, 'journal_authority_source'] = reference_matches.loc[manual_mask, 'journal_id'].map(manual_auth_src_map)
    reference_matches.loc[manual_mask, 'journal_final_id']   = reference_matches.loc[manual_mask, 'journal_background_source_id']
    reference_matches.loc[manual_mask, 'journal_final_name'] = reference_matches.loc[manual_mask, 'journal_name']
    reference_matches.loc[manual_mask, 'matched_authority_journal'] = True
    reference_matches.loc[manual_mask, 'journal_match_tier'] = np.where(
        reference_matches.loc[manual_mask, 'journal_background_source_id'].notna(),
        'manual_openalex_or_sciscinet_source', 'manual_authority_without_source_id')
    reference_matches.loc[manual_mask, 'match_method'] = 'manual_raw_to_authority'
    reference_matches.loc[manual_mask, 'match_score'] = 1.0
    reference_matches['has_final_journal'] = reference_matches['journal_final_id'].notna()
    print(f'manual mappings applied: {int(manual_mask.sum())} rows')
    reference_matches.to_parquet(REFERENCE_JOURNAL_MATCHES_PATH, index=False)

# ── Step 7: Diagnostics ───────────────────────────────────────────────────────
print('\n=== Step 7: Diagnostics ===')

unmatched_strings = (
    reference_matches[reference_matches['is_paper_candidate'] & ~reference_matches['has_final_journal'] & reference_matches['journal_raw'].notna()]
    .groupby('journal_raw', dropna=False).size().reset_index(name='reference_count')
    .sort_values('reference_count', ascending=False)
)
unmatched_strings.to_parquet(UNMATCHED_REFERENCE_JOURNAL_STRINGS_PATH, index=False)
unmatched_strings.to_csv(CSV_DIR / 'unmatched_journal_strings_requiring_review.csv', index=False)

matched_dedup = reference_matches[reference_matches['has_final_journal']][[
    'journal_raw','journal_final_id','journal_final_name','journal_match_tier',
    'journal_id','journal_background_source_id','journal_authority_source',
    'journal_name','match_method','match_score','matched_authority_journal','parser_used','parsed_journal_raw'
]].drop_duplicates().sort_values(['journal_match_tier','journal_final_name','journal_raw'])
matched_dedup.to_parquet(DEDUP_MATCHED_JOURNALS_PATH, index=False)
matched_dedup.to_csv(CSV_DIR / 'deduplicated_openalex_source_journal_matches.csv', index=False)

unmatched_dedup = reference_matches[reference_matches['is_paper_candidate'] & ~reference_matches['has_final_journal']][[
    'journal_raw','journal_name','journal_match_tier','match_method','parser_used','parsed_journal_raw'
]].drop_duplicates().sort_values(['journal_match_tier','match_method','journal_raw'])
unmatched_dedup.to_parquet(DEDUP_UNMATCHED_JOURNALS_PATH, index=False)
unmatched_dedup.to_csv(CSV_DIR / 'deduplicated_unmatched_or_unscoreable_journals.csv', index=False)

proposal_ref_summary = reference_matches.groupby([ID_COL, YEAR_COL], dropna=False).agg(
    parsed_reference_count=('reference_position', 'size'),
    paper_candidate_count=('is_paper_candidate', 'sum'),
    authority_journal_reference_count=('matched_authority_journal', 'sum'),
    scoreable_journal_reference_count=('has_final_journal', 'sum'),
    distinct_scoreable_journal_count=('journal_final_id', lambda x: x.dropna().astype(str).nunique()),
    no_scoreable_journal_paper_reference_count=('has_final_journal', lambda x: int((~x & reference_matches.loc[x.index, 'is_paper_candidate']).sum())),
).reset_index()
proposal_ref_summary['has_any_paper_reference_lacking_scoreable_journal'] = proposal_ref_summary['no_scoreable_journal_paper_reference_count'] > 0
proposal_ref_summary['has_two_or_more_distinct_scoreable_journals'] = proposal_ref_summary['distinct_scoreable_journal_count'] >= 2
proposal_ref_summary.to_parquet(PROPOSAL_MATCH_SUMMARY_PATH, index=False)
proposal_ref_summary.to_csv(CSV_DIR / 'proposal_reference_journal_match_summary.csv', index=False)

paper_refs = int(reference_matches['is_paper_candidate'].sum())
matched_refs = int((reference_matches['is_paper_candidate'] & reference_matches['matched_authority_journal']).sum())
scoreable_refs = int((reference_matches['is_paper_candidate'] & reference_matches['has_final_journal']).sum())
unmatched_refs = int((reference_matches['is_paper_candidate'] & ~reference_matches['has_final_journal']).sum())

print(f'likely paper refs:              {paper_refs:,}')
print(f'authority-matched paper refs:   {matched_refs:,}  ({100*matched_refs/max(paper_refs,1):.1f}%)')
print(f'scoreable (OpenAlex/SciSciNet): {scoreable_refs:,}  ({100*scoreable_refs/max(paper_refs,1):.1f}%)')
print(f'unmatched paper refs:           {unmatched_refs:,}  ({100*unmatched_refs/max(paper_refs,1):.1f}%)')
print(f'\nTop 30 unmatched journal strings:')
print(unmatched_strings.head(30).to_string(index=False))

# ── Step 8: Build proposal journal pairs ──────────────────────────────────────
print('\n=== Step 8: Build proposal journal pairs ===')

def build_proposal_journal_pairs(reference_matches):
    usable = reference_matches[reference_matches['has_final_journal']].copy()
    rows = []
    for (app_id, app_year), g in tqdm(usable.groupby([ID_COL, YEAR_COL], dropna=False), desc='Building pairs'):
        ids = sorted(g['journal_final_id'].dropna().astype(str).unique())
        total_pairs = len(ids) * (len(ids) - 1) // 2
        for j1, j2 in combinations(ids, 2):
            rows.append({ID_COL: app_id, YEAR_COL: app_year, 'journal_1': j1, 'journal_2': j2,
                         'proposal_distinct_journals': len(ids), 'proposal_total_pairs': total_pairs})
    return pd.DataFrame(rows)

proposal_pairs = build_proposal_journal_pairs(reference_matches)
proposal_pairs.to_parquet(PROPOSAL_JOURNAL_PAIRS_PATH, index=False)
proposal_pairs.to_csv(CSV_DIR / 'proposal_openalex_source_pairs.csv', index=False)

print(f'proposal source pairs: {len(proposal_pairs):,}')
print(f'proposals with ≥1 pair: {proposal_pairs[[ID_COL, YEAR_COL]].drop_duplicates().shape[0]:,}' if len(proposal_pairs) else 'proposals with ≥1 pair: 0')
print('\nDone. Output files:')
for f in sorted(OUTPUT_DIR.rglob('*')) + sorted(CSV_DIR.rglob('*')):
    if f.is_file():
        print(f'  {f.relative_to(WORK_DIR)}')
