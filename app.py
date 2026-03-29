"""
Cotton Options Pricing — Streamlit App
American Call + Put + Short-at-P  |  LSM Monte Carlo with 30-day AWP Lookback
"""

import streamlit as st
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cotton Options Pricer",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main container */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* Result card */
.result-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.result-card.call  { border-left: 5px solid #16a34a; }
.result-card.put   { border-left: 5px solid #dc2626; }
.result-card.short { border-left: 5px solid #7c3aed; }

.card-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.card-title.call  { color: #16a34a; }
.card-title.put   { color: #dc2626; }
.card-title.short { color: #7c3aed; }

.price-big {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.price-se {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.1rem;
}
.row-metrics {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 0.9rem;
}
.metric-chip {
    background: #fff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
    font-size: 0.8rem;
    color: #334155;
}
.metric-chip strong { color: #0f172a; }

/* ITM/OTM badge */
.badge-itm  { background:#dcfce7; color:#166534; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:600; }
.badge-otm  { background:#fee2e2; color:#991b1b; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:600; }

/* Summary table */
.summary-table { width:100%; border-collapse:collapse; font-size:0.88rem; }
.summary-table th { background:#f1f5f9; padding:8px 12px; text-align:left; border-bottom:2px solid #cbd5e1; }
.summary-table td { padding:8px 12px; border-bottom:1px solid #e2e8f0; }
.summary-table tr:last-child td { border-bottom:none; }

/* Divider */
.section-divider { border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0; }

/* Mobile tweaks */
@media (max-width: 640px) {
    .price-big { font-size: 1.6rem; }
    .row-metrics { gap: 0.4rem; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PRICING FUNCTIONS  (ported from notebook, no external deps beyond numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_awp_for_day(prices, current_day):
    if current_day <= 5:
        return prices[0]
    current_period = (current_day - 1) // 5
    start_day = (current_period - 1) * 5 + 1
    end_day   = min(start_day + 4, current_day - 1)
    return np.mean(prices[start_day:end_day + 1])


def calculate_modified_strike(stock_prices, ref_price, carry_rate,
                               days_elapsed, supplement=0.0):
    final_strike = ref_price + supplement + carry_rate * days_elapsed
    if days_elapsed == 0:
        return final_strike
    awp = calculate_awp_for_day(stock_prices, days_elapsed)
    return min(awp, final_strike)


def simulate_stock_prices_extended(S0, r, sigma, days_to_maturity,
                                   post_exercise_days, n_simulations):
    total_days = days_to_maturity + post_exercise_days
    dt         = 1 / 365.0
    Z          = np.random.normal(0, 1, (n_simulations, total_days))
    drift      = (r - 0.5 * sigma ** 2) * dt
    diffusion  = sigma * np.sqrt(dt)
    S          = S0 * np.exp(np.cumsum(drift + diffusion * Z, axis=1))
    return np.column_stack([np.full(n_simulations, S0), S])


def _run_lsm(S, exercise_values, r, days_to_maturity, n_simulations):
    dt = 1 / 365.0
    option_values       = exercise_values[:, -1].copy()
    exercise_indicators = np.zeros((n_simulations, days_to_maturity + 1), dtype=bool)
    exercise_indicators[:, -1] = option_values > 0

    for t in range(days_to_maturity - 1, 0, -1):
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
                     days_to_maturity, n_simulations, post_exercise_days,
                     r, direction):
    dt = 1 / 365.0
    final_payoffs     = np.zeros(n_simulations)
    original_payoffs  = np.zeros(n_simulations)
    lookback_benefits = np.zeros(n_simulations)
    exercise_times    = np.zeros(n_simulations)

    for i in range(n_simulations):
        ex_days = np.where(exercise_indicators[i])[0]
        if len(ex_days) == 0:
            continue
        ex_day  = ex_days[0]
        exercise_times[i] = ex_day
        k_ex = calculate_modified_strike(S[i, :ex_day + 1], ref_price,
                                         carry_rate, ex_day, supplement)

        if direction == 'call':
            orig_payoff = max(S[i, ex_day] - k_ex, 0)
        else:
            orig_payoff = max(k_ex - S[i, ex_day], 0)
        original_payoffs[i] = orig_payoff

        # 30-day lookback
        if direction == 'call':
            best = k_ex
            for offset in range(1, post_exercise_days + 1):
                mon = ex_day + offset
                if mon < S.shape[1]:
                    best = max(best, calculate_awp_for_day(S[i, :mon + 1], mon))
            lb = max(best - k_ex, 0)
        else:
            best = k_ex
            for offset in range(1, post_exercise_days + 1):
                mon = ex_day + offset
                if mon < S.shape[1]:
                    best = min(best, calculate_awp_for_day(S[i, :mon + 1], mon))
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


CARRY_RATE       = 0.0233
POST_EX_DAYS     = 30


def price_american_call(S0, loan_value, lrs, r, sigma,
                        days_to_maturity, n_simulations):
    S  = simulate_stock_prices_extended(S0, r, sigma, days_to_maturity,
                                        POST_EX_DAYS, n_simulations)
    ev = np.zeros((n_simulations, days_to_maturity + 1))
    for t in range(days_to_maturity + 1):
        for i in range(n_simulations):
            k        = calculate_modified_strike(S[i, :t + 1], loan_value,
                                                 CARRY_RATE, t, lrs)
            ev[i, t] = max(S[i, t] - k, 0)
    _, ei = _run_lsm(S, ev, r, days_to_maturity, n_simulations)

    # Moneyness at t=0
    k0 = calculate_modified_strike([S0], loan_value, CARRY_RATE, 0, lrs)
    intrinsic = max(S0 - k0, 0)
    moneyness = 'ITM' if S0 > k0 else ('ATM' if abs(S0 - k0) < 0.5 else 'OTM')

    res = _collect_payoffs(S, ei, loan_value, CARRY_RATE, lrs,
                           days_to_maturity, n_simulations, POST_EX_DAYS, r, 'call')
    res.update({'moneyness': moneyness, 'intrinsic': intrinsic,
                'strike_at_inception': k0})
    return res


def price_american_put(S0, loan_value, lrs, r, sigma,
                       days_to_maturity, n_simulations):
    S  = simulate_stock_prices_extended(S0, r, sigma, days_to_maturity,
                                        POST_EX_DAYS, n_simulations)
    ev = np.zeros((n_simulations, days_to_maturity + 1))
    for t in range(days_to_maturity + 1):
        for i in range(n_simulations):
            k        = calculate_modified_strike(S[i, :t + 1], loan_value,
                                                 CARRY_RATE, t, lrs)
            ev[i, t] = max(k - S[i, t], 0)
    _, ei = _run_lsm(S, ev, r, days_to_maturity, n_simulations)

    k0 = calculate_modified_strike([S0], loan_value, CARRY_RATE, 0, lrs)
    intrinsic = max(k0 - S0, 0)
    moneyness = 'ITM' if S0 < k0 else ('ATM' if abs(S0 - k0) < 0.5 else 'OTM')

    res = _collect_payoffs(S, ei, loan_value, CARRY_RATE, lrs,
                           days_to_maturity, n_simulations, POST_EX_DAYS, r, 'put')
    res.update({'moneyness': moneyness, 'intrinsic': intrinsic,
                'strike_at_inception': k0})
    return res


def price_put_at_short_entry(S0, short_entry_price, r, sigma,
                              days_to_maturity, n_simulations):
    S  = simulate_stock_prices_extended(S0, r, sigma, days_to_maturity,
                                        POST_EX_DAYS, n_simulations)
    ev = np.zeros((n_simulations, days_to_maturity + 1))
    for t in range(days_to_maturity + 1):
        for i in range(n_simulations):
            k        = calculate_modified_strike(S[i, :t + 1], short_entry_price,
                                                 CARRY_RATE, t, 0.0)
            ev[i, t] = max(k - S[i, t], 0)
    _, ei = _run_lsm(S, ev, r, days_to_maturity, n_simulations)

    intrinsic = max(short_entry_price - S0, 0)
    moneyness_label = ('ITM — currently in profit' if S0 < short_entry_price
                       else 'OTM — currently at a loss')
    moneyness_short = 'ITM' if S0 < short_entry_price else 'OTM'

    res = _collect_payoffs(S, ei, short_entry_price, CARRY_RATE, 0.0,
                           days_to_maturity, n_simulations, POST_EX_DAYS, r, 'put')
    res.update({'moneyness': moneyness_short,
                'moneyness_label': moneyness_label,
                'intrinsic': intrinsic,
                'unrealised_pnl': short_entry_price - S0})
    return res


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _badge(label):
    cls = 'badge-itm' if label in ('ITM', 'ITM — currently in profit') else 'badge-otm'
    return f'<span class="{cls}">{label}</span>'


def _chip(label, value):
    return f'<span class="metric-chip"><strong>{label}</strong>&nbsp;{value}</span>'


def render_result_card(title, color_cls, res, show_moneyness=True,
                       moneyness_override=None, extra_chips=None):
    moneyness_html = ''
    if show_moneyness:
        m = moneyness_override or res.get('moneyness', '')
        moneyness_html = '&nbsp;&nbsp;' + _badge(m)

    chips = [
        _chip('Std Error', f"±{res['standard_error']:.4f} ¢/lb"),
        _chip('Intrinsic', f"{res['intrinsic']:.4f} ¢/lb"),
        _chip('Avg Base Payoff', f"{res['avg_original_payoff']:.4f} ¢/lb"),
        _chip('Lookback Benefit', f"{res['avg_lookback_benefit']:.4f} ¢/lb"),
        _chip('Lookback Active', f"{res['lookback_activation_pct']:.1f}% of paths"),
        _chip('Exercise Rate', f"{res['exercise_rate_pct']:.1f}% of paths"),
    ]
    if extra_chips:
        chips.extend(extra_chips)

    chips_html = ''.join(chips)
    st.markdown(f"""
<div class="result-card {color_cls}">
  <div class="card-title {color_cls}">{title}{moneyness_html}</div>
  <div class="price-big">{res['option_price']:.4f} <span style="font-size:1rem;font-weight:500;color:#475569">¢/lb</span></div>
  <div class="price-se">Monte Carlo estimate ± {res['standard_error']:.4f}</div>
  <div class="row-metrics">{chips_html}</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("## 🌾 Cotton Options Pricer")
st.markdown(
    "American-style options with **30-day AWP lookback** — "
    "Longstaff-Schwartz Monte Carlo (LSM)"
)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("pricing_form"):
    st.markdown("### 📋 Market & Loan Parameters")

    col1, col2 = st.columns(2)
    with col1:
        S0 = st.number_input(
            "Current Cotton Price (¢/lb)",
            min_value=1.0, max_value=500.0, value=66.0, step=0.5,
            help="Spot price of cotton in cents per pound"
        )
        loan_value = st.number_input(
            "USDA Loan Rate (¢/lb)",
            min_value=1.0, max_value=200.0, value=52.0, step=0.5,
            help="USDA CCC loan rate — the base reference price"
        )
        lrs = st.number_input(
            "Loan Rate Supplement — LRS (¢/lb)",
            min_value=0.0, max_value=100.0, value=13.0, step=0.5,
            help="Additional supplement added on top of the loan rate"
        )

    with col2:
        r = st.number_input(
            "Risk-Free Rate (% p.a.)",
            min_value=0.0, max_value=30.0, value=5.0, step=0.25,
            format="%.2f",
            help="Annualised risk-free interest rate (e.g. 5 = 5%)"
        ) / 100.0

        sigma = st.number_input(
            "Volatility (% p.a.)",
            min_value=1.0, max_value=200.0, value=14.0, step=1.0,
            format="%.1f",
            help="Annualised cotton price volatility (e.g. 14 = 14%)"
        ) / 100.0

        days_to_maturity = st.number_input(
            "Days to Maturity",
            min_value=1, max_value=730, value=270, step=1,
            help="Number of calendar days until the option expires"
        )

    st.markdown("---")
    st.markdown("### ⚙️ Simulation Settings")

    n_sims_label = st.selectbox(
        "Monte Carlo Paths",
        options=["5,000  (fast — ~5s)", "10,000  (balanced — ~10s)", "20,000  (precise — ~20s)"],
        index=1,
        help="More paths = more accurate but slower. 10,000 is a good default."
    )
    n_simulations = int(n_sims_label.split(",")[0].replace(" ", "").replace(",", "")) * 1000 if "," in n_sims_label else int(n_sims_label.split()[0])
    # Parse cleanly
    n_simulations = {"5,000": 5000, "10,000": 10000, "20,000": 20000}[n_sims_label.split("(")[0].strip()]

    st.markdown("---")
    st.markdown("### 📉 Short Position (Optional)")
    st.caption("Fill this in only if you are short cotton and want to price your downside exposure from that entry level.")

    short_entry_str = st.text_input(
        "Short Entry Price P (¢/lb)",
        value="",
        placeholder="e.g. 70.00  — leave blank to skip",
        help="The price at which you entered your short position. Leave blank to skip this calculation."
    )

    st.markdown("")
    run_button = st.form_submit_button(
        "🚀  Run Pricing Model",
        use_container_width=True,
        type="primary"
    )

# ── Validation & execution ────────────────────────────────────────────────────
if run_button:
    # Parse optional short entry
    short_entry_price = None
    if short_entry_str.strip():
        try:
            short_entry_price = float(short_entry_str.strip())
            if short_entry_price <= 0:
                st.error("Short entry price must be greater than 0.")
                st.stop()
        except ValueError:
            st.error("Please enter a valid number for the Short Entry Price, or leave it blank.")
            st.stop()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📊 Pricing Results")

    # Parameter summary
    effective_strike = loan_value + lrs
    st.info(
        f"**Inputs:** S₀ = {S0:.2f} ¢/lb · Loan = {loan_value:.2f} · LRS = {lrs:.2f} · "
        f"Effective floor = {effective_strike:.2f} ¢/lb · σ = {sigma*100:.1f}% · "
        f"r = {r*100:.2f}% · T = {days_to_maturity}d · {n_simulations:,} paths"
    )

    # ── Run all pricers ───────────────────────────────────────────────────────
    progress = st.progress(0, text="Initialising simulation…")

    with st.spinner("Running Monte Carlo — this may take a few seconds…"):
        progress.progress(10, text="Pricing American Call…")
        call_res = price_american_call(S0, loan_value, lrs, r, sigma,
                                       days_to_maturity, n_simulations)

        progress.progress(45, text="Pricing American Put…")
        put_res = price_american_put(S0, loan_value, lrs, r, sigma,
                                     days_to_maturity, n_simulations)

        short_res = None
        if short_entry_price is not None:
            progress.progress(70, text="Pricing Short-at-P Put…")
            short_res = price_put_at_short_entry(S0, short_entry_price, r, sigma,
                                                  days_to_maturity, n_simulations)

        progress.progress(100, text="Done!")

    progress.empty()

    # ── Render result cards ───────────────────────────────────────────────────
    render_result_card(
        title="① American Call — Farmer / Long Holder Upside",
        color_cls="call",
        res=call_res,
    )
    st.caption(
        "Profits when the cotton price rises above the modified strike "
        "(min of AWP vs Loan+LRS+Carry). Includes 30-day post-exercise AWP lookback enhancement."
    )

    st.markdown("")
    render_result_card(
        title="② American Put — Cost of Downside Protection",
        color_cls="put",
        res=put_res,
    )
    st.caption(
        "Value of providing downside insurance against the cotton price falling below the "
        "modified strike. Includes 30-day AWP lookback for further decline after exercise."
    )

    if short_res is not None:
        st.markdown("")
        pnl = short_res['unrealised_pnl']
        pnl_chip = _chip(
            'Unrealised P&L',
            f"{'▲' if pnl >= 0 else '▼'} {abs(pnl):.2f} ¢/lb ({'profit' if pnl >= 0 else 'loss'})"
        )
        render_result_card(
            title=f"③ Short-at-P Put — Short entered at {short_entry_price:.2f} ¢/lb",
            color_cls="short",
            res=short_res,
            moneyness_override=short_res['moneyness_label'],
            extra_chips=[pnl_chip],
        )
        st.caption(
            f"Fair value of your downside exposure from your short entry at {short_entry_price:.2f} ¢/lb. "
            "Uses same AWP modified-strike framework but with P as the reference (no LRS — this is a market position)."
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 📋 Summary")

    rows = [
        ("American Call", "Farmer Upside",
         f"{call_res['option_price']:.4f}",
         f"±{call_res['standard_error']:.4f}",
         call_res['moneyness'],
         f"{call_res['intrinsic']:.4f}",
         f"{call_res['lookback_activation_pct']:.1f}%"),
        ("American Put", "Protection Cost",
         f"{put_res['option_price']:.4f}",
         f"±{put_res['standard_error']:.4f}",
         put_res['moneyness'],
         f"{put_res['intrinsic']:.4f}",
         f"{put_res['lookback_activation_pct']:.1f}%"),
    ]
    if short_res:
        rows.append((
            f"Put at P={short_entry_price:.2f}", "Short Exposure",
            f"{short_res['option_price']:.4f}",
            f"±{short_res['standard_error']:.4f}",
            short_res['moneyness'],
            f"{short_res['intrinsic']:.4f}",
            f"{short_res['lookback_activation_pct']:.1f}%",
        ))

    header = ("Option", "Role", "Price (¢/lb)", "Std Error", "Moneyness",
              "Intrinsic (¢/lb)", "Lookback Active")

    table_rows_html = ""
    for row in rows:
        cells = "".join(f"<td>{v}</td>" for v in row)
        table_rows_html += f"<tr>{cells}</tr>"

    header_html = "".join(f"<th>{h}</th>" for h in header)

    st.markdown(f"""
<table class="summary-table">
  <thead><tr>{header_html}</tr></thead>
  <tbody>{table_rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

    # ── Spread metric ─────────────────────────────────────────────────────────
    spread = call_res['option_price'] - put_res['option_price']
    st.markdown("")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Call − Put Spread",
                  f"{spread:+.4f} ¢/lb",
                  help="Positive = call more expensive than put (upside skew)")
    with col_b:
        st.metric("Lookback Boost (Call)",
                  f"{call_res['avg_lookback_benefit']:.4f} ¢/lb",
                  help="Average additional value from 30-day AWP lookback after call exercise")
    with col_c:
        st.metric("Lookback Boost (Put)",
                  f"{put_res['avg_lookback_benefit']:.4f} ¢/lb",
                  help="Average additional value from 30-day AWP lookback after put exercise")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.caption(
    "Model: Longstaff-Schwartz LSM · GBM price simulation · 30-day post-exercise AWP lookback · "
    "Modified strike = min(AWP, Loan+LRS+Carry) · Carry rate = 0.0233 ¢/lb/day"
)
