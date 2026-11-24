"""
Nanjing second-hand housing: price estimator (Gradio).

1. Run classproject.ipynb first. Its last cell saves model_artifacts.joblib.
2. Then run:   myenv\\Scripts\\python app.py
"""
import html
import random

import gradio as gr
import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402

# ------------------------------------------------------------------------------
# Load what the notebook saved
# ------------------------------------------------------------------------------
A = joblib.load('model_artifacts.joblib')
models = A['models']
feature_cols = A['feature_cols']
comm_map = A['comm_encoded_map']
global_mean = A['global_unit_mean']
district_mean = A['district_unit_mean']
data = A['data']
test = data[data['is_test']]
y_test = A['y_test']
test_preds = A['predictions']

# Model identity colors (validated for color-blind separation, light & dark)
MODEL_STYLE = {
    'Decision Tree (Baseline)':     {'short': 'Decision Tree',     'color': '#d97706', 'css': 'var(--c-dt)'},
    'Random Forest (Upgraded)':     {'short': 'Random Forest',     'color': '#2563eb', 'css': 'var(--c-rf)'},
    'Gradient Boosting (Upgraded)': {'short': 'Gradient Boosting', 'color': '#059669', 'css': 'var(--c-gb)'},
}
BEST = 'Gradient Boosting (Upgraded)'
BEST_MAE = mean_absolute_error(y_test, test_preds[BEST])
NO_COMPOUND = 'Not specified (use district average)'
INK = '#64748b'   # chart text: readable on light and dark backgrounds
GRID = '#94a3b8'


def options(col):
    return sorted(data[col].dropna().unique().tolist())


DISTRICTS = [d for d in options('district') if d != 'Other']
FLOOR_TIERS = options('floor_tier')
DECORATIONS = options('decoration')
BUILDING_TYPES = options('building_type')
ELEVATORS = options('elevator')
TENURES = options('tenure')
STRUCTURES = options('structure')
COMPOUNDS = [NO_COMPOUND] + sorted(comm_map.index.astype(str).tolist())
DISTRICT_COLORS = dict(zip(DISTRICTS, sns.color_palette('tab10', len(DISTRICTS)).as_hex()))

INPUT_KEYS = ['district', 'area', 'rooms', 'halls', 'baths', 'total_floors', 'floor_tier',
              'decoration', 'building_type', 'elevator', 'tenure', 'structure', 'is_south',
              'years_since_last_tx', 'compound']
DEFAULTS = ['Gulou', 90, 3, 2, 1, 18, 'Mid', 'Refined', 'Slab', 'Yes', 'Over_5_Yrs', 'Steel_Concrete',
            True, -1, NO_COMPOUND]


def money(v_10k):
    """10,000-RMB units → '¥4.14M'."""
    return f'¥{v_10k / 100:.2f}M'


def wan(v_10k):
    return f'{v_10k:,.0f} 万元'


# ------------------------------------------------------------------------------
# Chart styling
# ------------------------------------------------------------------------------
def new_fig(w, h):
    fig = Figure(figsize=(w, h), dpi=110)
    fig.patch.set_alpha(0)
    ax = fig.subplots()
    style_ax(ax)
    return fig, ax


def style_ax(ax):
    ax.set_facecolor('none')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_alpha(0.5)
    ax.tick_params(colors=INK, labelsize=9, length=0)
    ax.grid(color=GRID, alpha=0.2, lw=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def title(ax, text):
    ax.set_title(text, loc='left', color=INK, fontsize=11, fontweight='bold', pad=10)


# ------------------------------------------------------------------------------
# Feature row: built exactly like the notebook's training data
# ------------------------------------------------------------------------------
def build_row(v):
    area, rooms, halls, baths = float(v['area']), float(v['rooms']), float(v['halls']), float(v['baths'])
    row = {
        'area': area, 'rooms': rooms, 'halls': halls, 'baths': baths,
        'total_floors': float(v['total_floors']),
        'area_per_room': area / (rooms + 1),
        'total_rooms': rooms + halls + baths,
        'is_south': int(bool(v['is_south'])),
        'years_since_last_tx': float(v['years_since_last_tx']),
    }
    for col in ['district', 'decoration', 'building_type', 'elevator', 'floor_tier', 'tenure', 'structure']:
        row[f'{col}_{v[col]}'] = 1
    x = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0).astype(float)

    if v['compound'] in comm_map.index:
        est, source = float(comm_map[v['compound']]), f"compound “{v['compound']}”"
    else:
        est, source = float(district_mean.get(v['district'], global_mean)), f"{v['district']} district average"
    x['compound_est_unit_price'] = est
    x['compound_baseline_total'] = est * area / 10000.0
    return x, est, source


def same_inputs(a, b):
    if a is None or b is None or len(a) != len(b):
        return False
    for p, q in zip(a, b):
        if isinstance(p, (int, float, np.number)) and isinstance(q, (int, float, np.number)):
            if not np.isclose(float(p), float(q)):
                return False
        elif p != q:
            return False
    return True


# ------------------------------------------------------------------------------
# Tab 1: Price estimate
# ------------------------------------------------------------------------------
def error_badge(pred, actual):
    err = (pred - actual) / actual * 100
    level = 'good' if abs(err) < 5 else 'warn' if abs(err) < 15 else 'bad'
    return f'<span class="badge {level}">{err:+.1f}%</span>'


def render_result(v, preds, actual, est, source):
    best = preds[BEST]
    low, high = max(best - BEST_MAE, 0), best + BEST_MAE
    scale = max(list(preds.values()) + [actual or 0]) * 1.15

    rows = []
    for name, val in preds.items():
        s = MODEL_STYLE[name]
        marker = (f'<div class="actual-marker" style="left:{actual / scale * 100:.1f}%"></div>'
                  if actual is not None else '')
        badge = error_badge(val, actual) if actual is not None else ''
        rec = '<span class="tag">Recommended</span>' if name == BEST else ''
        rows.append(f'''
        <div class="model-row">
          <div class="model-name"><span class="dot" style="background:{s['css']}"></span>{s['short']}{rec}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{val / scale * 100:.1f}%;background:{s['css']}"></div>{marker}</div>
          <div class="model-val">{money(val)}{badge}</div>
        </div>''')

    if actual is not None:
        compare = f'''
        <div class="callout">
          <div><div class="label">Actual listed price</div><div class="callout-value">{money(actual)} <span class="muted">· {wan(actual)}</span></div></div>
          <div class="right"><div class="label">Estimate error</div><div class="callout-value">{error_badge(best, actual)}</div></div>
        </div>'''
        legend = '<div class="legend muted"><span class="legend-line"></span>Actual listed price</div>'
    else:
        compare = ''
        legend = ''

    details = (f"{html.escape(str(v['district']))} · {float(v['area']):g} m² · "
               f"{int(v['rooms'])} bed · {int(v['halls'])} living · {int(v['baths'])} bath · "
               f"{html.escape(str(v['decoration']))} decoration")
    return f'''
    <div class="result-card">
      <div class="label">Estimated price</div>
      <div class="result-price">{money(best)}</div>
      <div class="result-sub">{wan(best)} · {best * 10000 / float(v['area']):,.0f} RMB per m²</div>
      <div class="range">Likely range <b>{money(low)} – {money(high)}</b></div>
      <div class="muted small">{details}</div>
      {compare}
      <div class="divider"></div>
      <div class="label">Estimate by model</div>
      {''.join(rows)}
      {legend}
    </div>
    <details class="how">
      <summary>How is this calculated?</summary>
      <ul>
        <li>The estimate comes from a <b>Gradient Boosting</b> model trained on {len(data) - len(test):,} past Nanjing listings.
            On {len(test):,} listings it had never seen, its typical error was <b>±{money(BEST_MAE)}</b>; that is the “likely range”.</li>
        <li>The biggest factor is the <b>location price</b>. For this home it is <b>{est:,.0f} RMB/m²</b>, taken from the {html.escape(source)}.
            Choosing the exact residential compound gives a more accurate estimate.</li>
        <li>Floor area, layout, floor level, decoration and elevator then adjust the price up or down.</li>
      </ul>
    </details>'''


def predict(*args):
    *vals, state = args
    v = dict(zip(INPUT_KEYS, vals))
    if v['district'] is None or not v['area'] or v['rooms'] is None:
        return '<div class="result-card muted">Please fill in district, floor area and number of bedrooms.</div>'
    x, est, source = build_row(v)
    preds = {name: float(m.predict(x)[0]) for name, m in models.items()}
    actual = state['actual'] if state and same_inputs(state.get('inputs'), vals) else None
    return render_result(v, preds, actual, est, source)


def notice(text, kind='info'):
    return f'<div class="notice {kind}">{text}</div>'


DEFAULT_NOTICE = notice('Fill in the details below and select <b>Estimate price</b>. '
                        'You can also load a sample listing to see how accurate the estimate is.')


def sample_listing():
    r = test.sample(1).iloc[0]
    compound = r['小区名称'] if r['小区名称'] in comm_map.index else NO_COMPOUND
    vals = [r['district'] if r['district'] in DISTRICTS else DISTRICTS[0], float(r['area']), int(r['rooms']),
            int(r['halls']), int(r['baths']), int(r['total_floors']), r['floor_tier'], r['decoration'],
            r['building_type'], r['elevator'], r['tenure'], r['structure'], bool(r['is_south']),
            int(r['years_since_last_tx']), compound]
    msg = notice(f"Loaded real listing <b>#{r['id']}</b> ({html.escape(str(r['小区名称']))}). "
                 f"The model was not trained on this listing, so you can compare its estimate with the actual price.",
                 'success')
    return vals + [{'inputs': vals, 'actual': float(r['total_price'])}, msg]


def random_listing():
    rooms = random.randint(1, 5)
    vals = [
        random.choice(DISTRICTS),
        round(rooms * random.uniform(22, 38) + random.uniform(10, 30), 1),
        rooms,
        random.randint(1, 2),
        random.randint(1, min(3, rooms)),
        random.choice([6, 7, 11, 18, 24, 26, 33]),
        random.choice(['Low', 'Mid', 'High']),
        random.choice(['Refined', 'Simple', 'Roughcast', 'Other']),
        random.choice(['Slab', 'Tower', 'Slab-Tower']),
        random.choice(['Yes', 'No']),
        random.choice(['Over_5_Yrs', '2_to_5_Yrs', 'Under_2_Yrs']),
        'Steel_Concrete',
        random.random() < 0.85,
        -1,
        NO_COMPOUND,
    ]
    return vals + [None, notice('Generated a random home with realistic values. '
                                'It is not a real listing, so there is no actual price to compare.')]


def reset_form():
    return DEFAULTS + [None, DEFAULT_NOTICE]


# ------------------------------------------------------------------------------
# Tab 2: Market data
# ------------------------------------------------------------------------------
def stat_tiles(items):
    tiles = ''.join(f'<div class="stat"><div class="label">{k}</div><div class="stat-value">{v}</div></div>'
                    for k, v in items)
    return f'<div class="stat-grid">{tiles}</div>'


def explore(districts, area_min, area_max, price_max):
    d = data[data['district'].isin(districts or DISTRICTS)
             & data['area'].between(min(area_min, area_max), max(area_min, area_max))
             & (data['total_price'] <= price_max)]
    if d.empty:
        return notice('No listings match these filters. Try widening the area or price range.', 'warn'), None, None, None

    stats = stat_tiles([
        ('Listings', f'{len(d):,}'),
        ('Average price', money(d['total_price'].mean())),
        ('Median price', money(d['total_price'].median())),
        ('Average per m²', f"¥{d['unit_price'].mean():,.0f}"),
        ('Average size', f"{d['area'].mean():.0f} m²"),
    ])

    order = d.groupby('district')['unit_price'].median().sort_values(ascending=False).index.tolist()

    fig1, ax = new_fig(7, 4.6)
    for dist in order:
        s = d[d['district'] == dist]
        ax.scatter(s['area'], s['total_price'], s=7, alpha=0.45, color=DISTRICT_COLORS.get(dist, GRID),
                   label=dist, linewidths=0)
    ax.set_xlabel('Floor area (m²)')
    ax.set_ylabel('Total price (10,000 RMB)')
    title(ax, 'Price vs. floor area')
    ax.legend(markerscale=2.5, fontsize=8, frameon=False, labelcolor=INK)
    fig1.tight_layout()

    fig2, ax = new_fig(7, 4.6)
    sns.boxplot(data=d, y='district', x='unit_price', order=order, hue='district',
                palette=DISTRICT_COLORS, legend=False, ax=ax, fliersize=1, width=0.55,
                linewidth=1, boxprops=dict(alpha=0.85))
    style_ax(ax)
    ax.set_xlabel('Price per m² (RMB)')
    ax.set_ylabel('')
    title(ax, 'Price per m² by district (most expensive first)')
    fig2.tight_layout()

    table = (d[['id', '小区名称', 'district', 'area', 'rooms', 'halls', 'decoration', 'total_price', 'unit_price']]
             .sort_values('total_price', ascending=False).head(200)
             .rename(columns={'id': 'ID', '小区名称': 'Compound', 'district': 'District', 'area': 'Area (m²)',
                              'rooms': 'Bedrooms', 'halls': 'Living rooms', 'decoration': 'Decoration',
                              'total_price': 'Price (万元)', 'unit_price': 'RMB / m²'}))
    return stats, fig1, fig2, table


# ------------------------------------------------------------------------------
# Tab 3: Model accuracy
# ------------------------------------------------------------------------------
def score_cards():
    cards = []
    for name, s in MODEL_STYLE.items():
        p = test_preds[name]
        rec = '<span class="tag">Recommended</span>' if name == BEST else ''
        within10 = np.mean(np.abs(p - y_test) / y_test <= 0.10) * 100
        cards.append(f'''
        <div class="score-card" style="border-top:3px solid {s['css']}">
          <div class="score-head"><span class="dot" style="background:{s['css']}"></span>{s['short']}{rec}</div>
          <div class="score-main">{r2_score(y_test, p) * 100:.1f}%</div>
          <div class="muted small">of price differences explained (R² {r2_score(y_test, p):.3f})</div>
          <div class="score-rows">
            <div><span class="muted">Typical error</span><b>±{money(mean_absolute_error(y_test, p))}</b></div>
            <div><span class="muted">Within 10% of actual</span><b>{within10:.0f}% of homes</b></div>
            <div><span class="muted">RMSE</span><b>{money(np.sqrt(mean_squared_error(y_test, p)))}</b></div>
          </div>
        </div>''')
    return f'<div class="score-grid">{"".join(cards)}</div>'


def performance(model_name):
    pred = test_preds[model_name]
    color = MODEL_STYLE[model_name]['color']

    fig1, ax = new_fig(6.5, 4.6)
    ax.scatter(y_test, pred, s=7, alpha=0.35, color=color, linewidths=0)
    lim = max(y_test.max(), pred.max())
    ax.plot([0, lim], [0, lim], color=INK, ls='--', lw=1.2, label='Perfect estimate')
    ax.set_xlabel('Actual price (10,000 RMB)')
    ax.set_ylabel('Estimated price (10,000 RMB)')
    title(ax, 'Estimated vs. actual price (closer to the line is better)')
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig1.tight_layout()

    imp = pd.Series(models[model_name].feature_importances_, index=feature_cols).sort_values().tail(10)
    fig2, ax = new_fig(6.5, 4.6)
    ax.barh(imp.index, imp.values, color=color, height=0.6)
    ax.grid(axis='y', visible=False)
    title(ax, 'What the model relies on most')
    ax.set_xlabel('Importance')
    fig2.tight_layout()
    return fig1, fig2


# ------------------------------------------------------------------------------
# Look & feel: professional, neutral, high contrast
# ------------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    body_background_fill='#f8fafc',
    body_background_fill_dark='#0b1120',
    block_background_fill='#ffffff',
    block_background_fill_dark='#111827',
    block_border_width='1px',
    block_shadow='none',
    block_label_text_weight='600',
    button_primary_background_fill='#1e3a8a',
    button_primary_background_fill_hover='#1e40af',
    button_primary_background_fill_dark='#2563eb',
    button_primary_background_fill_hover_dark='#3b82f6',
    button_primary_text_color='#ffffff',
    button_secondary_background_fill='#ffffff',
    button_secondary_background_fill_hover='#f1f5f9',
    button_secondary_border_color='#cbd5e1',
    button_large_text_weight='600',
)

CSS = """
:root { --c-dt:#d97706; --c-rf:#2563eb; --c-gb:#059669; --navy:#1e3a8a; }
.dark { --c-rf:#3b82f6; --navy:#93c5fd; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* Header */
#header { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px;
          padding: 8px 4px 18px; border-bottom: 1px solid var(--border-color-primary); margin-bottom: 4px; }
#header h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -.02em; margin: 0; color: var(--body-text-color); }
#header p { margin: 4px 0 0; color: var(--body-text-color-subdued); font-size: .95rem; }
.kpis { display: flex; gap: 28px; }
.kpi .label { margin: 0; }
.kpi .v { font-size: 1.15rem; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--body-text-color); }

/* Generic */
.label { font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
         color: var(--body-text-color-subdued); margin-bottom: 4px; }
.muted { color: var(--body-text-color-subdued); }
.small { font-size: .86rem; }
.section-head { font-size: .95rem; font-weight: 700; color: var(--body-text-color); margin: 10px 0 0; }
.section-head span { display: inline-flex; width: 22px; height: 22px; border-radius: 50%; align-items: center;
                     justify-content: center; background: var(--navy); color: #fff; font-size: .75rem; margin-right: 8px; }
.dark .section-head span { color: #0b1120; }
.notice { padding: 12px 14px; border-radius: 8px; font-size: .9rem; border: 1px solid; line-height: 1.45; }
.notice.info    { background: rgba(37,99,235,.06);  border-color: rgba(37,99,235,.25); }
.notice.success { background: rgba(5,150,105,.07);  border-color: rgba(5,150,105,.3); }
.notice.warn    { background: rgba(217,119,6,.08);  border-color: rgba(217,119,6,.3); }

/* Result card */
.result-card { padding: 24px 26px; border-radius: 10px; background: var(--block-background-fill);
               border: 1px solid var(--border-color-primary); }
.result-price { font-size: 2.8rem; font-weight: 700; letter-spacing: -.03em; line-height: 1.1; color: var(--navy);
                font-variant-numeric: tabular-nums; }
.result-sub { font-size: 1rem; color: var(--body-text-color); margin-top: 4px; }
.range { display: inline-block; margin: 12px 0 10px; padding: 6px 12px; border-radius: 6px; font-size: .9rem;
         background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); }
.callout { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 16px;
           padding: 12px 14px; border-radius: 8px; background: var(--background-fill-secondary); }
.callout .right { text-align: right; }
.callout .label { margin: 0 0 2px; }
.callout-value { font-weight: 700; font-variant-numeric: tabular-nums; }
.divider { height: 1px; background: var(--border-color-primary); margin: 20px 0 14px; }

.model-row { display: grid; grid-template-columns: 210px 1fr 150px; gap: 14px; align-items: center; margin: 12px 0; }
.model-name { font-weight: 600; font-size: .9rem; display: flex; align-items: center; gap: 6px; }
.model-val { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }
.bar-track { position: relative; height: 8px; border-radius: 4px; background: var(--background-fill-secondary); }
.bar-fill { height: 100%; border-radius: 4px; }
.actual-marker { position: absolute; top: -5px; bottom: -5px; width: 2px; background: var(--body-text-color); }
.legend { font-size: .8rem; display: flex; align-items: center; gap: 8px; margin-top: 4px; }
.legend-line { display: inline-block; width: 2px; height: 14px; background: var(--body-text-color); }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
.tag { font-size: .68rem; font-weight: 600; padding: 1px 6px; border-radius: 4px; margin-left: 4px;
       border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued); }

.badge { display: inline-block; padding: 1px 7px; border-radius: 4px; font-size: .76rem; font-weight: 700; margin-left: 6px; }
.badge.good { background: rgba(5,150,105,.12); color: #047857; }
.badge.warn { background: rgba(217,119,6,.14); color: #b45309; }
.badge.bad  { background: rgba(220,38,38,.12); color: #b91c1c; }
.dark .badge.good { color: #34d399; } .dark .badge.warn { color: #fbbf24; } .dark .badge.bad { color: #f87171; }

.how { margin-top: 12px; padding: 12px 16px; border-radius: 10px; border: 1px solid var(--border-color-primary);
       background: var(--block-background-fill); font-size: .88rem; }
.how summary { cursor: pointer; font-weight: 600; }
.how ul { margin: 10px 0 0; padding-left: 18px; line-height: 1.55; }

/* Tiles */
.stat-grid, .score-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
.score-grid { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.stat, .score-card { padding: 16px 18px; border-radius: 10px; background: var(--block-background-fill);
                     border: 1px solid var(--border-color-primary); }
.stat-value { font-size: 1.45rem; font-weight: 700; letter-spacing: -.01em; font-variant-numeric: tabular-nums; }
.score-head { font-weight: 600; display: flex; align-items: center; gap: 6px; }
.score-main { font-size: 2rem; font-weight: 700; letter-spacing: -.02em; margin: 8px 0 0; font-variant-numeric: tabular-nums; }
.score-rows { margin-top: 12px; border-top: 1px solid var(--border-color-primary); padding-top: 8px; }
.score-rows div { display: flex; justify-content: space-between; padding: 4px 0; font-size: .88rem; }

@media (max-width: 720px) {
  .model-row { grid-template-columns: 1fr; gap: 6px; } .model-val { text-align: left; }
  #header h1 { font-size: 1.3rem; } .kpis { gap: 18px; }
}
"""

HEADER = f"""
<div id="header">
  <div>
    <h1>Nanjing Housing Price Estimator</h1>
    <p>Estimate the market price of a second-hand home from its location, size and features.</p>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="label">Listings analysed</div><div class="v">{len(data):,}</div></div>
    <div class="kpi"><div class="label">Accuracy (R²)</div><div class="v">{r2_score(y_test, test_preds[BEST]):.3f}</div></div>
    <div class="kpi"><div class="label">Typical error</div><div class="v">±{money(BEST_MAE)}</div></div>
  </div>
</div>
"""


def section(num, text):
    return gr.HTML(f'<div class="section-head"><span>{num}</span>{text}</div>')


# ------------------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------------------
with gr.Blocks(title='Nanjing Housing Price Estimator') as demo:
    gr.HTML(HEADER)

    with gr.Tabs():
        with gr.Tab('Price estimate'):
            state = gr.State(None)
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    info = gr.HTML(DEFAULT_NOTICE)

                    section(1, 'Location')
                    with gr.Group():
                        district = gr.Dropdown(DISTRICTS, value='Gulou', label='District',
                                               info='Administrative district in Nanjing')
                        compound = gr.Dropdown(COMPOUNDS, value=NO_COMPOUND, label='Residential compound (optional)',
                                               info='Type to search by Chinese name. Improves accuracy.')

                    section(2, 'Size & layout')
                    with gr.Group():
                        area = gr.Slider(15, 500, value=90, step=0.5, label='Floor area (m²)',
                                         info='Gross floor area as listed (建筑面积)')
                        with gr.Row():
                            rooms = gr.Number(3, label='Bedrooms', precision=0, minimum=0, maximum=10)
                            halls = gr.Number(2, label='Living rooms', precision=0, minimum=0, maximum=5)
                            baths = gr.Number(1, label='Bathrooms', precision=0, minimum=0, maximum=6)

                    section(3, 'Building & condition')
                    with gr.Group():
                        with gr.Row():
                            total_floors = gr.Number(18, label='Floors in building', precision=0, minimum=1, maximum=60)
                            floor_tier = gr.Dropdown(FLOOR_TIERS, value='Mid', label='Floor level',
                                                     info='Position within the building')
                        with gr.Row():
                            decoration = gr.Dropdown(DECORATIONS, value='Refined', label='Decoration',
                                                     info='Refined = fully finished, Roughcast = bare shell')
                            elevator = gr.Dropdown(ELEVATORS, value='Yes', label='Elevator')
                        is_south = gr.Checkbox(True, label='South-facing windows')
                    with gr.Accordion('Advanced details (optional)', open=False):
                        with gr.Row():
                            building_type = gr.Dropdown(BUILDING_TYPES, value='Slab', label='Building type')
                            structure = gr.Dropdown(STRUCTURES, value='Steel_Concrete', label='Structure')
                        with gr.Row():
                            tenure = gr.Dropdown(TENURES, value='Over_5_Yrs', label='Ownership period',
                                                 info='Affects transfer tax')
                            years_tx = gr.Number(-1, label='Years since last sale', precision=0,
                                                 info='Use -1 if unknown')

                    btn_predict = gr.Button('Estimate price', variant='primary', size='lg')
                    with gr.Row():
                        btn_sample = gr.Button('Load sample listing', variant='secondary', size='md')
                        btn_random = gr.Button('Random home', variant='secondary', size='md')
                        btn_reset = gr.Button('Reset', variant='secondary', size='md')

                with gr.Column(scale=6):
                    out_result = gr.HTML()

            inputs = [district, area, rooms, halls, baths, total_floors, floor_tier, decoration,
                      building_type, elevator, tenure, structure, is_south, years_tx, compound]
            btn_predict.click(predict, inputs + [state], out_result)
            btn_sample.click(sample_listing, None, inputs + [state, info]).then(predict, inputs + [state], out_result)
            btn_random.click(random_listing, None, inputs + [state, info]).then(predict, inputs + [state], out_result)
            btn_reset.click(reset_form, None, inputs + [state, info]).then(predict, inputs + [state], out_result)

        with gr.Tab('Market data'):
            with gr.Group():
                f_districts = gr.CheckboxGroup(DISTRICTS, value=DISTRICTS, label='Districts')
                with gr.Row():
                    f_amin = gr.Slider(10, 500, value=10, step=5, label='Minimum area (m²)')
                    f_amax = gr.Slider(10, 500, value=500, step=5, label='Maximum area (m²)')
                    f_pmax = gr.Slider(50, 3000, value=3000, step=50, label='Maximum price (万元)')
            e_stats = gr.HTML()
            with gr.Row():
                e_scatter = gr.Plot(show_label=False)
                e_box = gr.Plot(show_label=False)
            e_table = gr.Dataframe(label='Matching listings (200 most expensive)', interactive=False, max_height=420)
            e_inputs = [f_districts, f_amin, f_amax, f_pmax]
            for comp in e_inputs:
                comp.change(explore, e_inputs, [e_stats, e_scatter, e_box, e_table])

        with gr.Tab('Model accuracy'):
            gr.HTML(notice(f'All three models were tested on <b>{len(test):,} listings they had never seen</b>. '
                           'The recommended model is used for the estimate on the first tab.'))
            gr.HTML(score_cards())
            m_choice = gr.Radio(list(MODEL_STYLE), value=BEST, label='Show details for')
            with gr.Row():
                m_scatter = gr.Plot(show_label=False)
                m_imp = gr.Plot(show_label=False)
            m_choice.change(performance, m_choice, [m_scatter, m_imp])

    demo.load(explore, e_inputs, [e_stats, e_scatter, e_box, e_table])
    demo.load(performance, m_choice, [m_scatter, m_imp])
    demo.load(predict, inputs + [state], out_result)


if __name__ == '__main__':
    demo.launch(inbrowser=True, theme=THEME, css=CSS)
