import urllib.request
import json
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE = 'http://127.0.0.1:5000'

def test_get(path, expected_text=None):
    url = BASE + path
    try:
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200, f'Status {resp.status}'
            content = resp.read().decode('utf-8')
            if expected_text:
                assert expected_text in content, f'Expected {expected_text}'
            print(f'OK GET {path} (length {len(content)})')
            return content
    except Exception as e:
        print(f'FAIL GET {path}: {e}')
        sys.exit(1)

def test_post(path, data):
    url = BASE + path
    payload = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200, f'Status {resp.status}'
            res = json.loads(resp.read().decode('utf-8'))
            conseil = res.get('prediction', {}).get('conseil', 'N/A')
            risk = res.get('rapport_consequences', {}).get('niveau_risque', 'N/A')
            print(f'OK POST {path} -> {conseil} | Risque: {risk}')
            return res
    except Exception as e:
        print(f'FAIL POST {path}: {e}')
        sys.exit(1)

print('--- Test Routes HTTP ---')
test_get('/', 'PRAG')
test_get('/static/style.css', 'root')
test_get('/static/app.js', 'PRAG')

print('\n--- Test Health ---')
h = json.loads(test_get('/api/health'))
assert h['models_loaded'] is True, 'Models not loaded'

print('\n--- Test Predictions ---')
# Cacao Haut-Sassandra Mai (favorable)
test_post('/api/predict', {
    'region': 'Haut-Sassandra', 'culture': 'Cacao', 'type_sol': 'Argileux',
    'mois': 5, 'pluviometrie_mm': 190.0, 'temperature_celsius': 26.5, 'humidite_pct': 86.0
})

# Anacarde Poro Janvier (saison seche, non optimal pour semis)
test_post('/api/predict', {
    'region': 'Poro', 'culture': 'Anacarde', 'type_sol': 'Sablonneux',
    'mois': 1, 'pluviometrie_mm': 15.0, 'temperature_celsius': 33.5, 'humidite_pct': 42.0
})

# Mais Poro Juillet (saison des pluies dans le Nord)
test_post('/api/predict', {
    'region': 'Poro', 'culture': 'Maïs', 'type_sol': 'Limoneux',
    'mois': 7, 'pluviometrie_mm': 165.0, 'temperature_celsius': 28.0, 'humidite_pct': 76.0
})

print('\n--- Test History ---')
hist = json.loads(test_get('/api/history'))
print(f'OK Historique -> Sessions: {len(hist.get("sessions", []))}, Predictions: {len(hist.get("predictions", []))}')

print('\n--- Test Reload Models ---')
test_post('/api/reload-models', {})

print('\nALL TESTS PASSED SUCCESSFULLY!')
