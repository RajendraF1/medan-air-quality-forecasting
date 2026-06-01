import re

path = 'aplikasi_lab/templates/eval_dashboard.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the Horizon Tab Button
content = re.sub(
    r'<button onclick="switchTab\(\'horizon\'\)".*?Horizon Analysis.*?<\/button>',
    '',
    content,
    flags=re.DOTALL
)

# Remove the HORIZON TAB div panel
content = re.sub(
    r'<!-- ==================== HORIZON TAB ==================== -->.*?<div id="panel-horizon".*?<!-- ==================== EXPERIMENTS TAB ==================== -->',
    '<!-- ==================== EXPERIMENTS TAB ==================== -->',
    content,
    flags=re.DOTALL
)

# Remove JS switch tab trigger
content = content.replace("if (tab === 'horizon') loadHorizonAnalysis();", "")

# Remove JS function loadHorizonAnalysis()
content = re.sub(
    r'// ===== HORIZON ANALYSIS =====.*?async function loadHorizonAnalysis\(\) \{.*?\}(?=\s*// ===== HISTORY EXPERIMENT =====)',
    '',
    content,
    flags=re.DOTALL
)

# Remove 'Horizon' table header in experiments
content = re.sub(
    r'<th[^>]*>Horizon</th>',
    '',
    content,
    flags=re.DOTALL
)

# Remove horizon_prediction from history table row
content = re.sub(
    r'<td[^>]*>\$\{exp\.horizon_prediction \|\| \'--\'\}</td>',
    '',
    content,
    flags=re.DOTALL
)

# Replace '48 Jam' text if there's any chart title (Simulasi Prediksi 48 Jam -> 24 Jam)
content = content.replace('48 Jam', '24 Jam')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('HTML cleaned successfully.')
