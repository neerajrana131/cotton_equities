"""
Cotton Equity Options Pricer — Forward-Starting (Gin Date Aware)
American Call + Put + Short-at-P  |  LSM Monte Carlo  |  30-day AWP Lookback

Generic pricer that handles both scenarios in one model:
  • Gin Date = Today  →  standard equity (carry & AWP from day 0, user sets days to expiry)
  • Gin Date > Today  →  new-crop forward-starting equity (flat strike pre-gin, carry & AWP
                          begin only when bales are produced)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cotton Equity Pricer — Forward Starting",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top:1.5rem; padding-bottom:2rem; }
.result-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
               padding:1.2rem 1.4rem; margin-bottom:1rem; }
.result-card.call  { border-left:5px solid #16a34a; }
.result-card.put   { border-left:5px solid #dc2626; }
.result-card.short { border-left:5px solid #7c3aed; }
.card-title { font-size:1rem; font-weight:700; margin-bottom:0.6rem; }
.card-title.call  { color:#16a34a; }
.card-title.put   { color:#dc2626; }
.card-title.short { color:#7c3aed; }
.price-big { font-size:2rem; font-weight:800; letter-spacing:-0.5px; line-height:1.1; color:#0f172a; }
.price-se  { font-size:0.8rem; color:#64748b; margin-top:0.1rem; }
.row-metrics { display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.9rem; }
.metric-chip { background:#fff; border:1px solid #cbd5e1; border-radius:8px;
               padding:0.35rem 0.7rem; font-size:0.8rem; color:#334155; }
.metric-chip strong { color:#0f172a; }
.badge-itm { background:#dcfce7; color:#166534; border-radius:6px;
             padding:2px 8px; font-size:0.75rem; font-weight:600; }
.badge-otm { background:#fee2e2; color:#991b1b; border-radius:6px;
             padding:2px 8px; font-size:0.75rem; font-weight:600; }
.summary-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.summary-table th { background:#f1f5f9; padding:8px 12px; text-align:left;
                    border-bottom:2px solid #cbd5e1; }
.summary-table td { padding:8px 12px; border-bottom:1px solid #e2e8f0; }
.summary-table tr:last-child td { border-bottom:none; }
.section-divider { border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0; }
.timeline-bar { background:#f1f5f9; border:1px solid #cbd5e1; border-radius:10px;
                padding:0.75rem 1.2rem; margin:0.5rem 0 1rem 0;
                font-size:0.85rem; color:#334155; line-height:1.8; }
.timeline-bar strong { color:#0f172a; }
.tl-pre  { color:#0369a1; font-weight:600; }
.tl-post { color:#16a34a; font-weight:600; }
@media (max-width:640px) { .price-big { font-size:1.6rem; } }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
POST_EX_DAYS = 30
CARRY_RATE   = 0.0233   # ¢/lb/day
LB_LABEL     = "30-day"

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE PRICING FUNCTIONS  (gin_day-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_awp_since_gin(prices, t, gin_day):
    """
    5-day AWP computed only from gin_day onwards.

    • t <= gin_day : returns price at gin_day (no AWP before bales exist).
    • t >  gin_day : standard 5-day rolling average, with the clock starting
                     at gin_day instead of day 0.

    When gin_day = 0 this is identical to the original calculate_awp_for_day().
    """
    g = min(gin_day, len(prices) - 1)   # safe index cap
    if t <= g:
        return prices[g]
    days_since = t - g
    if days_since <= 5:
        return prices[g]
    cp = (days_since - 1) // 5
    s  = g + (cp - 1) * 5 + 1
    e  = min(g + cp * 5, t - 1)
    return np.mean(prices[s:e + 1])


def calculate_modified_strike(stock_prices, ref_price, carry_rate, t, gin_day, supplement=0.0):
    """
    Two-phase strike:
      Pre-gin  (t <= gin_day) : flat  = ref_price + supplement   (no carry, no AWP)
      Post-gin (t >  gin_day) : min(AWP_since_gin,
                                     ref_price + supplement + carry_rate * days_since_gin)

    When gin_day = 0 this collapses to the original modified-strike formula.
    """
    carry_days   = max(0, t - gin_day)
    final_strike = ref_price + supplement + carry_rate * carry_days
    if carry_days == 0:
        return final_strike
    awp = calculate_awp_since_gin(stock_prices, t, gin_day)
    return min(awp, final_strike)


def simulate_stock_prices_extended(S0, r, sigma, total_days, post_exercise_days, n_simulations):
    """GBM simulation over total_days + post_exercise_days steps."""
    dt        = 1 / 365.0
    steps     = total_days + post_exercise_days
    Z         = np.random.normal(0, 1, (n_simulations, steps))
    drift     = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    S         = S0 * np.exp(np.cumsum(drift + diffusion * Z, axis=1))
    return np.column_stack([np.full(n_simulations, S0), S])


def _run_lsm(S, exercise_values, r, total_days, n_simulations):
    """Longstaff-Schwartz backward induction — unchanged from original."""
    dt                  = 1 / 365.0
    option_values       = exercise_values[:, -1].copy()
    exercise_indicators = np.zeros((n_simulations, total_days + 1), dtype=bool)
    exercise_indicators[:, -1] = option_values > 0
    for t in range(total_days - 1, 0, -1):
        itm_mask = exercise_values[:, t] > 0
        if itm_mask.sum() > 2:
            X = S[itm_mask, t]
            Y = option_values[itm_mask] * np.exp(-r * dt)
            try:
                A    = np.vstack([np.ones_like(X), X, X ** 2]).T
                beta = np.linalg.lstsq(A, Y, rcond=None)[0]
                cont              = np.zeros(n_simulations)
                cont[itm_mask]    = np.dot(A, beta)
                exercise_now      = (exercise_values[:, t] > cont) & itm_mask
                option_values     = np.where(exercise_now,
                                             exercise_values[:, t],
                                             option_values * np.exp(-r * dt))
                exercise_indicators[:, t] = exercise_now
                for i in np.where(exercise_now)[0]:
                    exercise_indicators[i, t + 1:] = False
            except np.linalg.LinAlgError:
                option_values *= np.exp(-r * dt)
        else:
            option_values *= np.exp(-r * dt)
    return option_values, exercise_indicators


def _collect_payoffs(S, exercise_indicators, ref_price, carry_rate, supplement,
                     n_simulations, r, direction, gin_day):
    """
    Collect payoffs across simulations.  AWP lookback uses calculate_awp_since_gin
    so it only tracks prices after gin_day — consistent with how the strike was built.
    """
    dt                = 1 / 365.0
    final_payoffs     = np.zeros(n_simulations)
    original_payoffs  = np.zeros(n_simulations)
    lookback_benefits = np.zeros(n_simulations)
    exercise_times    = np.zeros(n_simulations)

    for i in range(n_simulations):
        ex_days = np.where(exercise_indicators[i])[0]
        if len(ex_days) == 0:
            continue
        ex_day = ex_days[0]
        exercise_times[i] = ex_day

        k_ex = calculate_modified_strike(
            S[i, :ex_day + 1], ref_price, carry_rate, ex_day, gin_day, supplement)

        orig_payoff = (max(S[i, ex_day] - k_ex, 0) if direction == 'call'
                       else max(k_ex - S[i, ex_day], 0))
        original_payoffs[i] = orig_payoff

        # 30-day post-exercise lookback — tracks AWP since gin
        best = k_ex
        for offset in range(1, POST_EX_DAYS + 1):
            mon = ex_day + offset
            if mon < S.shape[1]:
                best = min(best,
                           calculate_awp_since_gin(S[i, :mon + 1], mon, gin_day))
        lb = max(k_ex - best, 0)
        lookback_benefits[i] = lb
        final_payoffs[i]     = (orig_payoff + lb) * np.exp(-r * ex_day * dt)

    ex_mask = exercise_times > 0
    return {
        'option_price':            np.mean(final_payoffs),
        'standard_error':          np.std(final_payoffs) / np.sqrt(n_simulations),
        'avg_original_payoff':     np.mean(original_payoffs[ex_mask]) if ex_mask.any() else 0.0,
        'avg_lookback_benefit':    np.mean(lookback_benefits[ex_mask]) if ex_mask.any() else 0.0,
        'lookback_activation_pct': np.mean(lookback_benefits[ex_mask] > 0) * 100 if ex_mask.any() else 0.0,
        'exercise_rate_pct':       ex_mask.mean() * 100,
    }


def price_american_call(S0, loan_value, lrs, r, sigma, days_to_maturity, gin_day, n_simulations):
    total_days = gin_day + days_to_maturity
    S  = simulate_stock_prices_extended(S0, r, sigma, total_days, POST_EX_DAYS, n_simulations)
    ev = np.zeros((n_simulations, total_days + 1))

    # Pre-gin: flat strike, vectorised (no path dependency)
    if gin_day > 0:
        flat_k = loan_value + lrs
        ev[:, :gin_day] = np.maximum(S[:, :gin_day] - flat_k, 0)

    # Post-gin: AWP-modified strike (path-dependent, requires loop)
    for t in range(gin_day, total_days + 1):
        for i in range(n_simulations):
            k        = calculate_modified_strike(S[i, :t + 1], loan_value, CARRY_RATE, t, gin_day, lrs)
            ev[i, t] = max(S[i, t] - k, 0)

    _, ei = _run_lsm(S, ev, r, total_days, n_simulations)
    k0  = calculate_modified_strike([S0], loan_value, CARRY_RATE, 0, gin_day, lrs)
    res = _collect_payoffs(S, ei, loan_value, CARRY_RATE, lrs, n_simulations, r, 'call', gin_day)
    res.update({
        'moneyness':         'ITM' if S0 > k0 else ('ATM' if abs(S0 - k0) < 0.5 else 'OTM'),
        'intrinsic':         max(S0 - k0, 0),
        'strike_at_inception': k0,
    })
    return res


def price_american_put(S0, loan_value, lrs, r, sigma, days_to_maturity, gin_day, n_simulations):
    total_days = gin_day + days_to_maturity
    S  = simulate_stock_prices_extended(S0, r, sigma, total_days, POST_EX_DAYS, n_simulations)
    ev = np.zeros((n_simulations, total_days + 1))

    # Pre-gin: flat strike, vectorised
    if gin_day > 0:
        flat_k = loan_value + lrs
        ev[:, :gin_day] = np.maximum(flat_k - S[:, :gin_day], 0)

    # Post-gin: AWP-modified strike
    for t in range(gin_day, total_days + 1):
        for i in range(n_simulations):
            k        = calculate_modified_strike(S[i, :t + 1], loan_value, CARRY_RATE, t, gin_day, lrs)
            ev[i, t] = max(k - S[i, t], 0)

    _, ei = _run_lsm(S, ev, r, total_days, n_simulations)
    k0  = calculate_modified_strike([S0], loan_value, CARRY_RATE, 0, gin_day, lrs)
    res = _collect_payoffs(S, ei, loan_value, CARRY_RATE, lrs, n_simulations, r, 'put', gin_day)
    res.update({
        'moneyness':         'ITM' if S0 < k0 else ('ATM' if abs(S0 - k0) < 0.5 else 'OTM'),
        'intrinsic':         max(k0 - S0, 0),
        'strike_at_inception': k0,
    })
    return res


def price_put_at_short_entry(S0, short_entry_price, r, sigma, days_to_maturity, gin_day, n_simulations):
    total_days = gin_day + days_to_maturity
    S  = simulate_stock_prices_extended(S0, r, sigma, total_days, POST_EX_DAYS, n_simulations)
    ev = np.zeros((n_simulations, total_days + 1))

    # Pre-gin: flat strike = short entry price (no carry)
    if gin_day > 0:
        ev[:, :gin_day] = np.maximum(short_entry_price - S[:, :gin_day], 0)

    # Post-gin: carry on short entry price, gin-aware AWP
    for t in range(gin_day, total_days + 1):
        for i in range(n_simulations):
            k        = calculate_modified_strike(S[i, :t + 1], short_entry_price, CARRY_RATE, t, gin_day, 0.0)
            ev[i, t] = max(k - S[i, t], 0)

    _, ei = _run_lsm(S, ev, r, total_days, n_simulations)
    res   = _collect_payoffs(S, ei, short_entry_price, CARRY_RATE, 0.0, n_simulations, r, 'put', gin_day)
    res.update({
        'moneyness':       'ITM' if S0 < short_entry_price else 'OTM',
        'moneyness_label': ('ITM \u2014 currently in profit' if S0 < short_entry_price
                            else 'OTM \u2014 currently at a loss'),
        'intrinsic':       max(short_entry_price - S0, 0),
        'unrealised_pnl':  short_entry_price - S0,
    })
    return res


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART FUNCTIONS  (gin_day-aware)
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_BASE = dict(
    template='plotly_white', font=dict(family='Arial', size=12, color='#1e293b'),
    plot_bgcolor='#f8fafc', paper_bgcolor='#ffffff',
    margin=dict(l=50, r=20, t=50, b=50),
    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                xanchor='right', x=1, font=dict(size=11, color='#1e293b')),
)


def _add_gin_vline(fig, gin_day, gin_date_str):
    """Add a vertical marker at gin_day when it is > 0."""
    if gin_day > 0:
        fig.add_vline(
            x=gin_day,
            line_dash='dot', line_color='#0369a1', line_width=1.8,
            annotation_text=f'  Gin Date ({gin_date_str})',
            annotation_font_size=10, annotation_font_color='#0369a1',
            annotation_position='top right',
        )


def chart_fan(S0, r, sigma, total_days, loan, lrs, gin_day, gin_date_str):
    np.random.seed(42)
    n  = 400
    dt = 1 / 365.0
    Z  = np.random.normal(0, 1, (n, total_days))
    S  = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=1))
    S  = np.column_stack([np.full(n, S0), S])
    step = max(1, total_days // 60)
    days = list(range(0, total_days + 1, step))
    Ss   = S[:, ::step]
    p10, p25, p50 = (np.percentile(Ss, q, axis=0) for q in (10, 25, 50))
    p75, p90      = (np.percentile(Ss, q, axis=0) for q in (75, 90))

    # Floor: flat pre-gin, rising post-gin
    floor = [loan + lrs + CARRY_RATE * max(0, d - gin_day) for d in days]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days + days[::-1], y=list(p90) + list(p10[::-1]),
        fill='toself', fillcolor='rgba(56,189,248,0.22)',
        line=dict(color='rgba(0,0,0,0)'), name='10th\u201390th pct'))
    fig.add_trace(go.Scatter(x=days, y=p10,
        line=dict(color='rgba(14,165,233,0.55)', width=1.2),
        showlegend=False, hoverinfo='none'))
    fig.add_trace(go.Scatter(x=days, y=p90,
        line=dict(color='rgba(14,165,233,0.55)', width=1.2),
        showlegend=False, hoverinfo='none'))
    fig.add_trace(go.Scatter(x=days + days[::-1], y=list(p75) + list(p25[::-1]),
        fill='toself', fillcolor='rgba(14,165,233,0.48)',
        line=dict(color='rgba(0,0,0,0)'), name='25th\u201375th pct'))
    fig.add_trace(go.Scatter(x=days, y=p25,
        line=dict(color='rgba(3,105,161,0.75)', width=1.2),
        showlegend=False, hoverinfo='none'))
    fig.add_trace(go.Scatter(x=days, y=p75,
        line=dict(color='rgba(3,105,161,0.75)', width=1.2),
        showlegend=False, hoverinfo='none'))
    fig.add_trace(go.Scatter(x=days, y=p50,
        line=dict(color='#0c4a6e', width=3.5), name='Median path'))
    fig.add_trace(go.Scatter(x=days, y=floor,
        line=dict(color='#d97706', width=2.5, dash='dash'),
        name='Loan+LRS+Carry floor'))
    fig.add_hline(y=S0, line_dash='dot', line_color='#94a3b8', line_width=1,
                  annotation_text=f'  S\u2080={S0}\u00a2',
                  annotation_font_size=11, annotation_font_color='#94a3b8')
    _add_gin_vline(fig, gin_day, gin_date_str)
    fig.update_layout(**PLOTLY_BASE, height=360,
        xaxis=dict(title='Days from today',
                   title_font=dict(size=13, color='#1e293b'),
                   tickfont=dict(size=11, color='#1e293b')),
        yaxis=dict(title='Cotton price (\u00a2/lb)', ticksuffix='\u00a2',
                   title_font=dict(size=13, color='#1e293b'),
                   tickfont=dict(size=11, color='#1e293b')))
    st.plotly_chart(fig, use_container_width=True)
    gin_note = (f'  ·  Vertical dotted line = gin date (carry & AWP begin at day {gin_day})'
                if gin_day > 0 else '')
    st.caption(
        f"400 GBM simulations · Bands = 10th\u201390th (outer) and 25th\u201375th (inner) percentile · "
        f"Dashed amber = loan floor (flat pre-gin, rising at {CARRY_RATE}\u00a2/lb/day post-gin){gin_note}"
    )


def chart_strike_evolution(S0, r, sigma, total_days, loan, lrs, gin_day, gin_date_str):
    scenarios = [
        ('Bullish  (+15%)',       '#16a34a',  0.25,  11),
        ('Sideways (flat)',       '#0284c7',  0.00,  42),
        ('Bearish  (\u221210%)',  '#dc2626', -0.20,  77),
    ]
    step = max(1, total_days // 60)
    days = list(range(0, total_days + 1, step))

    fig          = go.Figure()
    n_scen       = len(scenarios)
    TRACES_PER   = 3
    DARK         = '#1e3a5f'
    ACTIVE       = '#2d6aa0'
    X_POS        = [0.0, 0.21, 0.43]

    for s_idx, (label, color, bias, seed) in enumerate(scenarios):
        np.random.seed(seed)
        dt   = 1 / 365.0
        Z    = np.random.normal(0, 1, total_days)
        path = [S0]
        for t in range(1, total_days + 1):
            path.append(path[-1] * np.exp(
                (r - 0.5 * sigma ** 2 + bias) * dt + sigma * np.sqrt(dt) * Z[t - 1]))

        prices_l, awps_l, strikes_l = [], [], []
        for t in days:
            carry_days = max(0, t - gin_day)
            fl         = loan + lrs + CARRY_RATE * carry_days
            if t == 0:
                awp = S0
            else:
                awp = calculate_awp_since_gin(path[:t + 1], t, gin_day)
            strike = min(awp, fl) if carry_days > 0 else fl
            prices_l.append(path[t])
            awps_l.append(awp)
            strikes_l.append(strike)

        vis = (s_idx == 1)
        fig.add_trace(go.Scatter(x=days, y=prices_l,
            line=dict(color=color, width=1.2, dash='dot'), visible=vis,
            name='Spot price',
            hovertemplate='Day %{x}<br>Spot: %{y:.2f}\u00a2<extra></extra>'))
        fig.add_trace(go.Scatter(x=days, y=awps_l,
            line=dict(color=color, width=1.5, dash='dashdot'), visible=vis,
            name='AWP (since gin)',
            hovertemplate='Day %{x}<br>AWP: %{y:.2f}\u00a2<extra></extra>'))
        fig.add_trace(go.Scatter(x=days, y=strikes_l,
            line=dict(color=color, width=3), visible=vis,
            name='Modified strike',
            hovertemplate='Day %{x}<br>Strike: %{y:.2f}\u00a2<extra></extra>'))

    # Floor — always visible
    floor_vals = [loan + lrs + CARRY_RATE * max(0, d - gin_day) for d in days]
    fig.add_trace(go.Scatter(x=days, y=floor_vals,
        line=dict(color='#d97706', width=2, dash='dash'),
        name='Loan+LRS+Carry floor', visible=True,
        hovertemplate='Day %{x}<br>Floor: %{y:.2f}\u00a2<extra></extra>'))

    def vis_list(active):
        v = []
        for i in range(n_scen):
            v += [i == active] * TRACES_PER
        v.append(True)
        return v

    updatemenus = []
    for i, (label, color, bias, seed) in enumerate(scenarios):
        clean = label.strip()
        bgcolor_update = {
            'updatemenus[0].bgcolor': ACTIVE if i == 0 else DARK,
            'updatemenus[1].bgcolor': ACTIVE if i == 1 else DARK,
            'updatemenus[2].bgcolor': ACTIVE if i == 2 else DARK,
        }
        updatemenus.append(dict(
            type='buttons',
            x=X_POS[i], y=1.28, xanchor='left',
            showactive=False,
            bgcolor=ACTIVE if i == 1 else DARK,
            bordercolor='#0369a1',
            font=dict(size=11, color='white'),
            buttons=[dict(
                method='update',
                args=[{'visible': vis_list(i)}, bgcolor_update],
                label=clean,
            )],
        ))

    _add_gin_vline(fig, gin_day, gin_date_str)
    fig.update_layout(
        **PLOTLY_BASE, height=420,
        xaxis=dict(title='Days from today',
                   title_font=dict(size=13, color='#1e293b'),
                   tickfont=dict(size=11, color='#1e293b')),
        yaxis=dict(title='Price (\u00a2/lb)', ticksuffix='\u00a2',
                   title_font=dict(size=13, color='#1e293b'),
                   tickfont=dict(size=11, color='#1e293b')),
        updatemenus=updatemenus,
        margin=dict(l=50, r=20, t=90, b=50),
    )
    fig.add_hline(y=S0, line_dash='dot', line_color='#94a3b8', line_width=1,
                  annotation_text=f'  S\u2080={S0}\u00a2',
                  annotation_font_size=11, annotation_font_color='#94a3b8')
    st.plotly_chart(fig, use_container_width=True)
    pre_note = (f'  ·  Pre-gin strike is flat at Loan+LRS (day 0\u2013{gin_day})'
                if gin_day > 0 else '')
    st.caption(
        "Thick line = modified strike = min(AWP\u2099, Loan+LRS+Carry)  \u00b7  "
        "Dotted = spot  \u00b7  Dash-dot = AWP (5-day, since gin)  \u00b7  "
        f"Dashed amber = carry floor{pre_note}  \u00b7  Click buttons to switch scenario"
    )


def chart_lookback_breakdown(call_res, put_res):
    labels = ['American Call', 'American Put']
    base   = [call_res['avg_original_payoff'], put_res['avg_original_payoff']]
    lb     = [call_res['avg_lookback_benefit'], put_res['avg_lookback_benefit']]
    totals = [b + l for b, l in zip(base, lb)]
    lb_pct = [l / t * 100 if t > 0 else 0 for l, t in zip(lb, totals)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Base payoff (at exercise)',
        x=labels, y=base,
        marker_color=['#16a34a', '#dc2626'],
        text=[f'{v:.4f}\u00a2' for v in base],
        textposition='inside', textfont=dict(color='white', size=12),
        hovertemplate='%{x}<br>Base payoff: %{y:.4f}\u00a2/lb<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name=f'{LB_LABEL} lookback benefit',
        x=labels, y=lb,
        marker_color=['#4ade80', '#f87171'],
        text=[f'+{v:.4f}\u00a2\n({p:.1f}%)' for v, p in zip(lb, lb_pct)],
        textposition='inside', textfont=dict(color='#1e293b', size=11),
        hovertemplate='%{x}<br>Lookback benefit: %{y:.4f}\u00a2/lb<extra></extra>',
    ))
    for i, (lbl, total) in enumerate(zip(labels, totals)):
        fig.add_annotation(x=lbl, y=total, text=f'<b>{total:.4f}\u00a2 total</b>',
                           showarrow=False, yshift=12, font=dict(size=12, color='#1e293b'))
    fig.update_layout(**PLOTLY_BASE, height=340, barmode='stack',
        xaxis=dict(tickfont=dict(size=13, color='#1e293b')),
        yaxis=dict(title='Value (\u00a2/lb)', ticksuffix='\u00a2',
                   title_font=dict(size=13, color='#1e293b'),
                   tickfont=dict(size=11, color='#1e293b')))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"The **{LB_LABEL} lookback window** contributes "
        f"**{lb_pct[0]:.1f}%** of the total call price and "
        f"**{lb_pct[1]:.1f}%** of the total put price."
    )


def run_sensitivity_analysis(S0, loan, lrs, r, sigma, days_to_maturity, gin_day, n_sens=300):
    results = {'spot': [], 'vol': []}
    for offset in [-9, -6, -3, 0, 3, 6, 9]:
        s = max(1.0, S0 + offset)
        c = price_american_call(s, loan, lrs, r, sigma, days_to_maturity, gin_day, n_sens)
        p = price_american_put(s, loan, lrs, r, sigma, days_to_maturity, gin_day, n_sens)
        results['spot'].append({
            'spot': f'{s:.0f}\u00a2', 'offset': f'{offset:+d}\u00a2',
            'call': c['option_price'], 'put': p['option_price'],
            'call_se': c['standard_error'], 'put_se': p['standard_error'],
            'moneyness': 'ITM' if s > c['strike_at_inception'] else 'OTM',
            'is_base': offset == 0,
        })
    for v in [0.08, 0.10, 0.12, sigma, 0.17, 0.20, 0.25]:
        c = price_american_call(S0, loan, lrs, r, v, days_to_maturity, gin_day, n_sens)
        p = price_american_put(S0, loan, lrs, r, v, days_to_maturity, gin_day, n_sens)
        results['vol'].append({
            'vol': f'{v * 100:.0f}%', 'call': c['option_price'], 'put': p['option_price'],
            'call_se': c['standard_error'], 'put_se': p['standard_error'],
            'is_base': abs(v - sigma) < 0.001,
        })
    return results


def chart_sensitivity(sens):
    tab_spot, tab_vol = st.tabs(["\U0001f4cd vs Spot Price", "\U0001f4c9 vs Volatility"])

    with tab_spot:
        rows  = sens['spot']
        spots = [r['spot']  for r in rows]
        calls = [r['call']  for r in rows]
        puts  = [r['put']   for r in rows]
        fig   = go.Figure()
        fig.add_trace(go.Bar(name='Call price', x=spots, y=calls,
            marker_color=['#16a34a' if r['is_base'] else '#4ade80' for r in rows],
            text=[f'{v:.3f}' for v in calls], textposition='outside',
            textfont=dict(size=10),
            hovertemplate='Spot=%{x}<br>Call: %{y:.4f}\u00a2/lb<extra></extra>'))
        fig.add_trace(go.Bar(name='Put price', x=spots, y=puts,
            marker_color=['#dc2626' if r['is_base'] else '#f87171' for r in rows],
            text=[f'{v:.3f}' for v in puts], textposition='outside',
            textfont=dict(size=10),
            hovertemplate='Spot=%{x}<br>Put: %{y:.4f}\u00a2/lb<extra></extra>'))
        base_row = next(r for r in rows if r['is_base'])
        fig.add_annotation(x=base_row['spot'],
            y=max(base_row['call'], base_row['put']) + 0.5,
            text='BASE', showarrow=False, font=dict(size=10, color='#475569'),
            bgcolor='#f1f5f9', borderpad=3)
        fig.update_layout(**PLOTLY_BASE, height=340, barmode='group',
            xaxis=dict(title='Spot price',
                       title_font=dict(size=13, color='#1e293b'),
                       tickfont=dict(size=11, color='#1e293b')),
            yaxis=dict(title='Option price (\u00a2/lb)', ticksuffix='\u00a2',
                       title_font=dict(size=13, color='#1e293b'),
                       tickfont=dict(size=11, color='#1e293b')))
        st.plotly_chart(fig, use_container_width=True)

        hdr = ['Spot', 'Call (\u00a2/lb)', '\u00b1SE', 'Put (\u00a2/lb)', '\u00b1SE', '']
        rh  = ''
        for row in rows:
            bg  = 'background:#f0fdf4;border-left:3px solid #16a34a;' if row['is_base'] else ''
            itm = 'itm' if row['moneyness'] == 'ITM' else 'otm'
            rh += (f'<tr style="{bg}"><td>{row["spot"]} '
                   f'<small style="color:#94a3b8">({row["offset"]})</small></td>'
                   f'<td style="color:#16a34a;font-weight:600">{row["call"]:.4f}</td>'
                   f'<td style="color:#94a3b8;font-size:.85em">\u00b1{row["call_se"]:.4f}</td>'
                   f'<td style="color:#dc2626;font-weight:600">{row["put"]:.4f}</td>'
                   f'<td style="color:#94a3b8;font-size:.85em">\u00b1{row["put_se"]:.4f}</td>'
                   f'<td><span class="badge-{itm}">{row["moneyness"]}</span></td></tr>')
        st.markdown(
            f'<table class="summary-table"><thead><tr>'
            + ''.join(f'<th>{h}</th>' for h in hdr)
            + f'</tr></thead><tbody>{rh}</tbody></table>',
            unsafe_allow_html=True)

    with tab_vol:
        rows  = sens['vol']
        vols  = [r['vol']  for r in rows]
        calls = [r['call'] for r in rows]
        puts  = [r['put']  for r in rows]
        fig   = go.Figure()
        fig.add_trace(go.Scatter(name='Call price', x=vols, y=calls,
            mode='lines+markers', line=dict(color='#16a34a', width=2.5),
            marker=dict(size=[11 if r['is_base'] else 7 for r in rows],
                        color='#16a34a',
                        symbol=['star' if r['is_base'] else 'circle' for r in rows]),
            hovertemplate='Vol=%{x}<br>Call: %{y:.4f}\u00a2/lb<extra></extra>'))
        fig.add_trace(go.Scatter(name='Put price', x=vols, y=puts,
            mode='lines+markers', line=dict(color='#dc2626', width=2.5),
            marker=dict(size=[11 if r['is_base'] else 7 for r in rows],
                        color='#dc2626',
                        symbol=['star' if r['is_base'] else 'circle' for r in rows]),
            hovertemplate='Vol=%{x}<br>Put: %{y:.4f}\u00a2/lb<extra></extra>'))
        base_v = next(r for r in rows if r['is_base'])
        fig.add_annotation(x=base_v['vol'],
            y=max(base_v['call'], base_v['put']) + 0.5,
            text='BASE', showarrow=False, font=dict(size=10, color='#475569'),
            bgcolor='#f1f5f9', borderpad=3)
        fig.update_layout(**PLOTLY_BASE, height=320,
            xaxis=dict(title='Volatility',
                       title_font=dict(size=13, color='#1e293b'),
                       tickfont=dict(size=11, color='#1e293b')),
            yaxis=dict(title='Option price (\u00a2/lb)', ticksuffix='\u00a2',
                       title_font=dict(size=13, color='#1e293b'),
                       tickfont=dict(size=11, color='#1e293b')))
        st.plotly_chart(fig, use_container_width=True)

        hdr = ['Volatility', 'Call (\u00a2/lb)', '\u00b1SE', 'Put (\u00a2/lb)', '\u00b1SE']
        rh  = ''
        for row in rows:
            bg  = 'background:#f0fdf4;border-left:3px solid #16a34a;' if row['is_base'] else ''
            tag = '&nbsp;<small style="color:#94a3b8">(base)</small>' if row['is_base'] else ''
            rh += (f'<tr style="{bg}"><td style="font-weight:{"700" if row["is_base"] else "400"}">'
                   f'{row["vol"]}{tag}</td>'
                   f'<td style="color:#16a34a;font-weight:600">{row["call"]:.4f}</td>'
                   f'<td style="color:#94a3b8;font-size:.85em">\u00b1{row["call_se"]:.4f}</td>'
                   f'<td style="color:#dc2626;font-weight:600">{row["put"]:.4f}</td>'
                   f'<td style="color:#94a3b8;font-size:.85em">\u00b1{row["put_se"]:.4f}</td>'
                   f'</tr>')
        st.markdown(
            f'<table class="summary-table"><thead><tr>'
            + ''.join(f'<th>{h}</th>' for h in hdr)
            + f'</tr></thead><tbody>{rh}</tbody></table>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _badge(label):
    cls = 'badge-itm' if label in ('ITM', 'ITM \u2014 currently in profit') else 'badge-otm'
    return f'<span class="{cls}">{label}</span>'

def _chip(label, value):
    return f'<span class="metric-chip"><strong>{label}</strong>&nbsp;{value}</span>'

def render_result_card(title, color_cls, res, moneyness_override=None, extra_chips=None):
    m              = moneyness_override or res.get('moneyness', '')
    moneyness_html = '&nbsp;&nbsp;' + _badge(m) if m else ''
    chips = [
        _chip('Std Error',       f"\u00b1{res['standard_error']:.4f} \u00a2/lb"),
        _chip('Intrinsic',       f"{res['intrinsic']:.4f} \u00a2/lb"),
        _chip('Avg Base Payoff', f"{res['avg_original_payoff']:.4f} \u00a2/lb"),
        _chip(f'{LB_LABEL} LB Benefit', f"{res['avg_lookback_benefit']:.4f} \u00a2/lb"),
        _chip('LB Active',       f"{res['lookback_activation_pct']:.1f}% of paths"),
        _chip('Exercise Rate',   f"{res['exercise_rate_pct']:.1f}% of paths"),
    ]
    if extra_chips:
        chips.extend(extra_chips)
    chips_html = ''.join(chips)
    st.markdown(f"""
<div class="result-card {color_cls}">
  <div class="card-title {color_cls}">{title}{moneyness_html}</div>
  <div class="price-big">{res['option_price']:.4f}
    <span style="font-size:1rem;font-weight:500;color:#475569">\u00a2/lb</span>
  </div>
  <div class="price-se">Monte Carlo estimate \u00b1 {res['standard_error']:.4f}</div>
  <div class="row-metrics">{chips_html}</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════
for _k, _v in [('call_res', None), ('put_res', None), ('short_res', None),
               ('params', None), ('sens_res', None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT — HEADER + FORM
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🌾 Cotton Equity Options Pricer — Forward Starting")
st.markdown(
    "American-style options with **30-day AWP lookback** — Longstaff-Schwartz Monte Carlo (LSM)  \n"
    "Carry and AWP begin at the **Gin Date**. "
    "Set Gin Date = today for a standard current-crop equity."
)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

today = date.today()

with st.form("pricing_form"):
    st.markdown("### \U0001f4cb Market & Loan Parameters")
    col1, col2 = st.columns(2)
    with col1:
        S0         = st.number_input("Futures Price (\u00a2/lb)",
                        min_value=1.0, max_value=500.0, value=66.0, step=0.5,
                        help="Current futures price in cents per pound")
        loan_value = st.number_input("USDA Loan Rate (\u00a2/lb)",
                        min_value=1.0, max_value=200.0, value=52.0, step=0.5,
                        help="USDA CCC loan rate — the base reference price")
        lrs        = st.number_input("Loan Rate Supplement \u2014 LRS (\u00a2/lb)",
                        min_value=0.0, max_value=100.0, value=13.0, step=0.5,
                        help="Supplement added on top of the loan rate")
    with col2:
        r     = st.number_input("Risk-Free Rate (% p.a.)",
                    min_value=0.0, max_value=30.0, value=5.0, step=0.25,
                    format="%.2f", help="Annualised risk-free rate") / 100.0
        sigma = st.number_input("Volatility (% p.a.)",
                    min_value=1.0, max_value=200.0, value=14.0, step=1.0,
                    format="%.1f", help="Annualised cotton price volatility") / 100.0
        gin_date = st.date_input(
            "Gin Date",
            value=today,
            min_value=today,
            help="Date when bales are produced and carry/AWP begin. "
                 "Set to today for a standard current-crop equity.")

    st.markdown("---")
    st.markdown("### \u23f1\ufe0f Option Expiry")
    days_to_maturity = st.number_input(
        "Days to Expiry (from Gin Date)",
        min_value=1, max_value=730, value=270, step=1,
        help="Calendar days from gin date to option expiry. Default 270. "
             "For new crop: expiry = gin date + this value. "
             "For current crop (gin = today): expiry = today + this value.")

    st.markdown("---")
    st.markdown("### \u2699\ufe0f Simulation Settings")
    n_sims_label = st.selectbox("Monte Carlo Paths", options=[
        "5,000    (quick check \u2014 ~9s at T=270d)",
        "10,000   (balanced \u2014 ~18s at T=270d)",
        "20,000   (recommended \u2014 ~37s at T=270d)",
        "100,000  (high precision \u2014 ~3 min at T=270d)",
    ], index=2,
    help="20,000 recommended for most uses. 100,000 for final quotes.")
    raw = n_sims_label.split()[0].replace(",", "")
    n_simulations = int(raw)

    st.markdown("---")
    st.markdown("### \U0001f4c9 Short Position (Optional)")
    st.caption("Fill in only if you are short cotton and want to price your downside exposure.")
    short_entry_str = st.text_input(
        "Short Entry Price P (\u00a2/lb)", value="",
        placeholder="e.g. 70.00  \u2014 leave blank to skip")

    st.markdown("")
    run_button = st.form_submit_button(
        "\U0001f680  Run Pricing Model", use_container_width=True, type="primary")


# ── Derived timing values ──────────────────────────────────────────────────────
gin_day    = max(0, (gin_date - today).days)
total_days = gin_day + days_to_maturity
expiry_date = gin_date + timedelta(days=days_to_maturity)
gin_date_str = gin_date.strftime('%b %d, %Y')

# ── Timeline banner ────────────────────────────────────────────────────────────
if gin_day > 0:
    st.markdown(f"""
<div class="timeline-bar">
📅 <strong>Today</strong> ({today.strftime('%b %d')})
&nbsp; <span class="tl-pre">──── {gin_day} days · flat strike, no carry, no AWP ────▶</span>
&nbsp; 🏭 <strong>Gin Date</strong> ({gin_date.strftime('%b %d, %Y')})
&nbsp; <span class="tl-post">──── {days_to_maturity} days · carry + AWP begin ────▶</span>
&nbsp; ⏱ <strong>Expiry</strong> ({expiry_date.strftime('%b %d, %Y')})
&nbsp;·&nbsp; <strong>Total window: {total_days} days</strong>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown(f"""
<div class="timeline-bar">
🏭 <strong>Gin Date = Today</strong> ({today.strftime('%b %d, %Y')})
&nbsp; <span class="tl-post">──── {days_to_maturity} days · carry + AWP from day 0 ────▶</span>
&nbsp; ⏱ <strong>Expiry</strong> ({expiry_date.strftime('%b %d, %Y')})
</div>
""", unsafe_allow_html=True)


# ── Run pricing ────────────────────────────────────────────────────────────────
if run_button:
    short_entry_price = None
    if short_entry_str.strip():
        try:
            short_entry_price = float(short_entry_str.strip())
            if short_entry_price <= 0:
                st.error("Short entry price must be > 0.")
                st.stop()
        except ValueError:
            st.error("Please enter a valid number for Short Entry Price, or leave blank.")
            st.stop()

    prog = st.progress(0, text="Initialising\u2026")
    with st.spinner("Running Monte Carlo\u2026"):
        prog.progress(10, text="Pricing American Call\u2026")
        call_res = price_american_call(
            S0, loan_value, lrs, r, sigma, days_to_maturity, gin_day, n_simulations)
        prog.progress(48, text="Pricing American Put\u2026")
        put_res  = price_american_put(
            S0, loan_value, lrs, r, sigma, days_to_maturity, gin_day, n_simulations)
        short_res = None
        if short_entry_price is not None:
            prog.progress(76, text="Pricing Short-at-P Put\u2026")
            short_res = price_put_at_short_entry(
                S0, short_entry_price, r, sigma, days_to_maturity, gin_day, n_simulations)
        prog.progress(100, text="Done!")
    prog.empty()

    st.session_state.call_res  = call_res
    st.session_state.put_res   = put_res
    st.session_state.short_res = short_res
    st.session_state.sens_res  = None
    st.session_state.params    = dict(
        S0=S0, loan_value=loan_value, lrs=lrs, r=r, sigma=sigma,
        days_to_maturity=days_to_maturity, gin_day=gin_day,
        total_days=total_days, n_simulations=n_simulations,
        short_entry_price=short_entry_price,
        gin_date_str=gin_date_str, expiry_date_str=expiry_date.strftime('%b %d, %Y'),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULTS + VISUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.call_res is not None:
    p  = st.session_state.params
    cr = st.session_state.call_res
    pr = st.session_state.put_res
    sr = st.session_state.short_res

    # ── Pricing results ────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### \U0001f4ca Pricing Results")

    gin_info = (f"  ·  Gin = {p['gin_date_str']}  ·  Pre-gin = {p['gin_day']}d"
                if p['gin_day'] > 0 else "  ·  Gin = Today")
    st.info(
        f"**S\u2080** = {p['S0']:.2f}\u00a2  \u00b7  Loan = {p['loan_value']:.2f}  \u00b7  "
        f"LRS = {p['lrs']:.2f}  \u00b7  Floor = {p['loan_value']+p['lrs']:.2f}\u00a2  \u00b7  "
        f"\u03c3 = {p['sigma']*100:.1f}%  \u00b7  r = {p['r']*100:.2f}%  \u00b7  "
        f"Post-gin T = {p['days_to_maturity']}d  \u00b7  Total = {p['total_days']}d  \u00b7  "
        f"Expiry = {p['expiry_date_str']}  \u00b7  {p['n_simulations']:,} paths{gin_info}"
    )

    render_result_card("① American Call — Farmer / Long Holder Upside", "call", cr)
    st.caption("Profits when cotton rises above the modified strike. Includes 30-day AWP lookback.")
    st.markdown("")
    render_result_card("② American Put — Cost of Downside Protection", "put", pr)
    st.caption("Cost of downside insurance. Includes 30-day AWP lookback.")

    if sr is not None:
        st.markdown("")
        sep      = p.get('short_entry_price') or sr['intrinsic'] + p['S0']
        pnl      = sr['unrealised_pnl']
        pnl_chip = _chip('Unrealised P&L',
                         f"{'▲' if pnl >= 0 else '▼'} {abs(pnl):.2f}\u00a2/lb "
                         f"({'profit' if pnl >= 0 else 'loss'})")
        render_result_card(
            f"③ Short-at-P Put — Short entered at {sep:.2f}\u00a2",
            "short", sr, moneyness_override=sr['moneyness_label'],
            extra_chips=[pnl_chip])
        st.caption("Fair value of downside exposure from short entry. 30-day AWP lookback.")

    # ── Summary table ──────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### \U0001f4cb Summary")
    tbl_rows = [
        ("American Call", "Farmer Upside",
         f"{cr['option_price']:.4f}", f"\u00b1{cr['standard_error']:.4f}",
         cr['moneyness'], f"{cr['intrinsic']:.4f}",
         f"{cr['lookback_activation_pct']:.1f}%"),
        ("American Put", "Protection Cost",
         f"{pr['option_price']:.4f}", f"\u00b1{pr['standard_error']:.4f}",
         pr['moneyness'], f"{pr['intrinsic']:.4f}",
         f"{pr['lookback_activation_pct']:.1f}%"),
    ]
    if sr:
        tbl_rows.append(("Put at Short Entry", "Short Exposure",
            f"{sr['option_price']:.4f}", f"\u00b1{sr['standard_error']:.4f}",
            sr['moneyness'], f"{sr['intrinsic']:.4f}",
            f"{sr['lookback_activation_pct']:.1f}%"))
    hdr = ("Option", "Role", "Price (\u00a2/lb)", "Std Error",
           "Moneyness", "Intrinsic (\u00a2/lb)", "LB Active")
    hh  = ''.join(f'<th>{h}</th>' for h in hdr)
    rh  = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>'
                  for row in tbl_rows)
    st.markdown(
        f'<table class="summary-table"><thead><tr>{hh}</tr></thead>'
        f'<tbody>{rh}</tbody></table>',
        unsafe_allow_html=True)
    st.markdown("")
    ca, cb, cc = st.columns(3)
    spread = cr['option_price'] - pr['option_price']
    with ca: st.metric("Call \u2212 Put Spread", f"{spread:+.4f} \u00a2/lb")
    with cb: st.metric("LB Boost (Call)",        f"{cr['avg_lookback_benefit']:.4f} \u00a2/lb")
    with cc: st.metric("LB Boost (Put)",         f"{pr['avg_lookback_benefit']:.4f} \u00a2/lb")

    # ── Visual analysis ────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### \U0001f4c8 Visual Analysis")
    st.caption("Click any section to expand.")

    with st.expander("\U0001f4c8  Monte Carlo Fan Chart — Range of possible price paths"):
        chart_fan(p['S0'], p['r'], p['sigma'], p['total_days'],
                  p['loan_value'], p['lrs'], p['gin_day'], p['gin_date_str'])

    with st.expander("\U0001f4c9  Modified Strike Evolution — How the effective strike moves over time"):
        chart_strike_evolution(p['S0'], p['r'], p['sigma'], p['total_days'],
                               p['loan_value'], p['lrs'], p['gin_day'], p['gin_date_str'])

    with st.expander("\U0001f9e9  Lookback Benefit Breakdown — Base payoff vs lookback enhancement"):
        chart_lookback_breakdown(cr, pr)

    with st.expander("\u26a1  Sensitivity Analysis — How prices respond to input changes"):
        st.caption(
            "Runs the LSM model at 300 paths across 7 spot prices and 7 vol levels. "
            "Takes approximately 10\u201315 seconds. (300 paths is sufficient for trend analysis.)"
        )
        if st.button("\u25b6\ufe0f  Run Sensitivity Analysis", type="secondary"):
            with st.spinner("Running 14 sensitivity scenarios\u2026"):
                st.session_state.sens_res = run_sensitivity_analysis(
                    p['S0'], p['loan_value'], p['lrs'],
                    p['r'], p['sigma'], p['days_to_maturity'],
                    p['gin_day'], n_sens=300)
        if st.session_state.sens_res is not None:
            chart_sensitivity(st.session_state.sens_res)
        else:
            st.info("Press **▶ Run Sensitivity Analysis** above to generate charts and tables.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.caption(
    "Cotton Equity Options Pricer — Forward Starting  \u00b7  "
    "LSM Monte Carlo  \u00b7  30-day AWP Lookback  \u00b7  "
    "Carry & AWP begin at Gin Date  \u00b7  "
    "Gin Date = today \u21d2 standard current-crop equity"
)
