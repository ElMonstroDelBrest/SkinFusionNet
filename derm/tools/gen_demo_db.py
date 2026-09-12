"""Build a demo audit.db (+ images) for the SkinFusionNet tablet app from
external_val/per_image.csv. Pre-fills patients / analyses / verified labels so
the app is well-populated for a demo. Output: /tmp/demo/audit.db + /tmp/demo/images/."""
import csv, json, os, sqlite3, random, glob
from datetime import datetime, timedelta
from PIL import Image
from derm import paths
from derm.export.model_manifest import validate_manifest
try:
    import pillow_heif; pillow_heif.register_heif_opener()
except Exception as e:
    print('heif:', e)

random.seed(7)
SRC = str(paths.EXTERNAL_VAL)
OUT = '/tmp/demo'
IMG = f'{OUT}/images'
os.makedirs(IMG, exist_ok=True)
for f in glob.glob(f'{IMG}/*'):
    os.remove(f)
DB = f'{OUT}/audit.db'
if os.path.exists(DB): os.remove(DB)

MANIFEST = validate_manifest(
    paths.APP_MODELS / 'model_manifest.json', paths.APP_MODELS
)
MODEL = MANIFEST['bundle_id']
MODEL_SHA = MANIFEST['bundle_sha256']
PREPROCESSING = MANIFEST['preprocessing']['version']
DECISION_POLICY = MANIFEST['decision_policy']['version']
APPDIR = '/data/user/0/com.stage.skinfusionnet/app_flutter/images'
THRESHOLDS = MANIFEST['decision_policy']['thresholds']
THR_MEL = float(THRESHOLDS['melanoma'])
THR_ATYP = float(THRESHOLDS['atypical'])
THR_NEV = float(THRESHOLDS['nevus'])
NOW = datetime(2026, 6, 11)

SURNAMES = ['Bennett','Carter','Dawson','Ellis','Foster','Grant','Hayes','Irving',
            'Jensen','Keller','Lawson','Mercer','Novak','Owens','Porter','Quinn',
            'Reyes','Sutton','Turner','Underwood','Vance','Walsh','Xu','Yates','Zimmer',
            'Abbott','Brooks','Chen','Dunn','Espinoza']
INITIALS = list('ABCDEFGHJKLMNPRSTW')

def conv(folder, fn, out_id):
    p = os.path.join(SRC, folder, fn)
    im = Image.open(p).convert('RGB')
    w, h = im.size; m = max(w, h)
    if m > 1400: im = im.resize((int(w*1400/m), int(h*1400/m)))
    im.save(f'{IMG}/{out_id}.jpg', quality=88)

rows = list(csv.DictReader(open(f'{SRC}/per_image.csv')))
random.shuffle(rows)
print('rows:', len(rows))

# group into patients of 1..5 images
patients = []
i = 0
pid_n = 0
while i < len(rows):
    k = random.choice([1, 2, 2, 3, 3, 4, 5])
    patients.append(rows[i:i+k])
    i += k
    pid_n += 1
print('patients:', len(patients))

con = sqlite3.connect(DB); c = con.cursor()
c.executescript('''
CREATE TABLE android_metadata (locale TEXT);
INSERT INTO android_metadata VALUES ('en_US');
CREATE TABLE patient(id TEXT PRIMARY KEY, name TEXT NOT NULL, sex INTEGER,
  sun_exposure INTEGER, antecedents TEXT, created_ts INTEGER, updated_ts INTEGER);
CREATE TABLE analysis(id TEXT PRIMARY KEY, ts INTEGER NOT NULL, patient_id TEXT,
  lesion_id TEXT, capture_date INTEGER, image_hash TEXT, image_path TEXT,
  model_version TEXT, model_bundle_sha256 TEXT, preprocessing_version TEXT,
  decision_policy_version TEXT, pred_class INTEGER, probs TEXT, confidence REAL, features TEXT,
  score_mel REAL, score_atyp REAL, score_nevus REAL, thr_mel REAL, thr_atyp REAL,
  thr_nevus REAL, verdict INTEGER, non_benign INTEGER, quality_flag INTEGER,
  archived INTEGER DEFAULT 0);
CREATE TABLE verified_label(id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL,
  ts INTEGER, verified_class INTEGER, source TEXT, note TEXT);
CREATE TABLE validation(id INTEGER PRIMARY KEY AUTOINCREMENT, analysis_id TEXT NOT NULL,
  ts INTEGER, md_agree INTEGER, md_class INTEGER);
CREATE TABLE audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER,
  event TEXT, analysis_id TEXT, detail TEXT);
CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT);
''')

aidx = 0; n_ver = 0
ANTS = [None, None, 'Family history of melanoma', 'Multiple nevi', 'Sun-damaged skin',
        'Previous excision', 'Fair skin (phototype II)']
for pi, group in enumerate(patients):
    pid = f'p_demo{pi:03d}'
    name = f'{random.choice(SURNAMES)}, {random.choice(INITIALS)}.'
    sex = random.randint(0, 1)
    sun = random.randint(0, 3)
    ant = random.choice(ANTS)
    window = random.randint(45, 650)
    start = NOW - timedelta(days=window)
    created = int(start.timestamp() * 1000)
    gdays = sorted(random.randint(0, window) for _ in range(len(group)))
    c.execute('INSERT INTO patient VALUES (?,?,?,?,?,?,?)',
              (pid, name, sex, sun, ant, created, created))
    for j, r in enumerate(group):
        aidx += 1
        aid = f'a_demo{aidx:03d}'
        conv(r['folder'], r['file'], aid)
        pn, pm, pa = float(r['p_nevus']), float(r['p_mel']), float(r['p_atyp'])
        argmax = int(r['argmax']); cov = float(r['coverage'])
        probs = [pn, pm, pa]
        verdict = 2 if pm >= THR_MEL else (1 if pa >= THR_ATYP else 0)
        non_benign = 1 if (pm >= THR_MEL or pa >= THR_ATYP) else 0
        qflag = 1 if cov < 0.05 else 0
        cap = start + timedelta(days=gdays[j])
        capms = int(cap.timestamp() * 1000)
        c.execute('''INSERT INTO analysis(
                  id,ts,patient_id,lesion_id,capture_date,image_hash,image_path,
                  model_version,model_bundle_sha256,preprocessing_version,
                  decision_policy_version,pred_class,probs,confidence,features,
                  score_mel,score_atyp,score_nevus,thr_mel,thr_atyp,thr_nevus,
                  verdict,non_benign,quality_flag,archived)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                  (aid, capms, pid, None, capms, f'{abs(hash(aid)):016x}'[:16],
                   f'{APPDIR}/{aid}.jpg', MODEL, MODEL_SHA, PREPROCESSING,
                   DECISION_POLICY, argmax, json.dumps(probs), max(probs), '[]',
                   pm, pa, pn, THR_MEL, THR_ATYP, THR_NEV, verdict, non_benign,
                   qflag, 0))
        # ground truth: Nevi->0, Melanom->1. Verify ~92% (leave some for "To verify").
        if aidx % 13 != 0:
            vc = 1 if r['folder'] == 'Melanom' else 0
            c.execute('INSERT INTO verified_label(analysis_id,ts,verified_class,source,note) VALUES (?,?,?,?,?)',
                      (aid, capms, vc, 'biopsy', None))
            n_ver += 1

c.execute('INSERT INTO settings(key,value) VALUES (?,?)',
          (f'thresholds:{DECISION_POLICY}',
           json.dumps({'mel': THR_MEL, 'atyp': THR_ATYP, 'nevus': THR_NEV})))
c.execute('PRAGMA user_version = 5')
con.commit()
print(f'analyses: {aidx}  verified: {n_ver}  images: {len(glob.glob(IMG+"/*.jpg"))}')
print('db size:', os.path.getsize(DB), 'bytes')
con.close()
