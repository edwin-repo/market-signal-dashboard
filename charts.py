#!/usr/bin/env python3
"""
Plotly chart builders for backtesting visualization.
All charts use the dashboard's dark theme colors.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Theme colors — Obsidian Glass
BG = '#07070d'
BG2 = '#0f0f18'
TEXT = '#f0f0f5'
GREEN = '#3fb950'
RED = '#ff6b6b'
YELLOW = '#e3b341'
BLUE = '#58a6ff'
GRAY = '#8888a0'

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TEXT, family='Inter, sans-serif', size=12),
    xaxis=dict(gridcolor='#16162a', zerolinecolor='#16162a'),
    yaxis=dict(gridcolor='#16162a', zerolinecolor='#16162a'),
    margin=dict(l=60, r=30, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
    hovermode='x unified',
)


def spy_overlay_chart(spy_df: pd.DataFrame, signal_dates: list,
                      signals_df: pd.DataFrame = None,
                      title: str = 'Signal Overlay on SPY',
                      horizon: str = '3m') -> go.Figure:
    """SPY price chart with signal markers colored by outcome."""
    fig = go.Figure()

    # SPY price line
    fig.add_trace(go.Scatter(
        x=spy_df['date'], y=spy_df['close'],
        mode='lines', name=spy_label,
        line=dict(color=BLUE, width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>' + spy_label + ': $%{y:.2f}<extra></extra>'
    ))

    # Signal markers
    if signals_df is not None and not signals_df.empty:
        fwd_col = f'fwd_{horizon}'
        if fwd_col in signals_df.columns:
            wins = signals_df[signals_df[fwd_col] > 0]
            losses = signals_df[signals_df[fwd_col] <= 0]
            no_data = signals_df[signals_df[fwd_col].isna()]

            if not wins.empty:
                fig.add_trace(go.Scatter(
                    x=wins['date'], y=wins['close'],
                    mode='markers', name=f'Win ({horizon})',
                    marker=dict(color=GREEN, size=8, symbol='triangle-up',
                                line=dict(color='white', width=1)),
                    hovertemplate='%{x|%Y-%m-%d}<br>SPY: $%{y:.2f}<br>Win<extra></extra>'
                ))
            if not losses.empty:
                fig.add_trace(go.Scatter(
                    x=losses['date'], y=losses['close'],
                    mode='markers', name=f'Loss ({horizon})',
                    marker=dict(color=RED, size=8, symbol='triangle-down',
                                line=dict(color='white', width=1)),
                    hovertemplate='%{x|%Y-%m-%d}<br>SPY: $%{y:.2f}<br>Loss<extra></extra>'
                ))
            if not no_data.empty:
                fig.add_trace(go.Scatter(
                    x=no_data['date'], y=no_data['close'],
                    mode='markers', name='Pending',
                    marker=dict(color=GRAY, size=6, symbol='circle'),
                    hovertemplate='%{x|%Y-%m-%d}<br>SPY: $%{y:.2f}<br>Pending<extra></extra>'
                ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title='SPY Price ($)',
        height=500,
        **LAYOUT_DEFAULTS,
    )
    return fig


def putcall_chart(putcall_df: pd.DataFrame, spy_df: pd.DataFrame = None,
                  threshold_10d: float = 0.70, threshold_30d: float = 0.65) -> go.Figure:
    """Put/Call ratio with 10d/30d MAs and SPY on secondary axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Raw ratio (faded)
    fig.add_trace(go.Scatter(
        x=putcall_df['date'], y=putcall_df['equity_pc_ratio'],
        mode='lines', name='Daily Ratio',
        line=dict(color=GRAY, width=0.5),
        opacity=0.3,
    ), secondary_y=False)

    # 10-day MA
    fig.add_trace(go.Scatter(
        x=putcall_df['date'], y=putcall_df['pc_10d_ma'],
        mode='lines', name='10d MA',
        line=dict(color=YELLOW, width=2),
    ), secondary_y=False)

    # 30-day MA
    fig.add_trace(go.Scatter(
        x=putcall_df['date'], y=putcall_df['pc_30d_ma'],
        mode='lines', name='30d MA',
        line=dict(color='#bc8cff', width=2),
    ), secondary_y=False)

    # Threshold lines
    fig.add_hline(y=threshold_10d, line_dash='dash', line_color=YELLOW,
                  annotation_text=f'10d threshold: {threshold_10d}',
                  annotation_font_color=YELLOW, opacity=0.5)
    fig.add_hline(y=threshold_30d, line_dash='dash', line_color='#bc8cff',
                  annotation_text=f'30d threshold: {threshold_30d}',
                  annotation_font_color='#bc8cff', opacity=0.5)

    # SPY on secondary axis
    if spy_df is not None:
        pc_dates = set(putcall_df['date'])
        spy_filtered = spy_df[spy_df['date'].isin(pc_dates)]
        fig.add_trace(go.Scatter(
            x=spy_filtered['date'], y=spy_filtered['close'],
            mode='lines', name=spy_label,
            line=dict(color=BLUE, width=1),
            opacity=0.6,
        ), secondary_y=True)

    fig.update_layout(
        title=dict(text='CBOE Equity Put/Call Ratio', font=dict(size=16)),
        height=450,
        **LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(title_text='Put/Call Ratio', secondary_y=False, gridcolor='#16162a')
    fig.update_yaxes(title_text='SPY ($)', secondary_y=True, gridcolor='#16162a')
    return fig


def sector_strength_chart(sectors_df: pd.DataFrame,
                          ratio_col: str = 'igv_spy_ratio',
                          etf_name: str = 'IGV',
                          lookback: int = 60) -> go.Figure:
    """Sector/SPY relative strength with breakdown zones highlighted."""
    df = sectors_df.copy()
    df['rolling_min'] = df[ratio_col].rolling(lookback).min()
    df['rolling_max'] = df[ratio_col].rolling(lookback).max()
    df['pctile'] = (df[ratio_col] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'] + 1e-10)

    fig = go.Figure()

    # Relative strength line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df[ratio_col],
        mode='lines', name=f'{etf_name}/SPY Ratio',
        line=dict(color=BLUE, width=1.5),
    ))

    # Highlight "puking" zones (below 10th percentile)
    puke_zones = df[df['pctile'] < 0.10]
    if not puke_zones.empty:
        fig.add_trace(go.Scatter(
            x=puke_zones['date'], y=puke_zones[ratio_col],
            mode='markers', name='Breakdown (<10th pctile)',
            marker=dict(color=RED, size=4, opacity=0.6),
        ))

    fig.update_layout(
        title=dict(text=f'{etf_name} vs SPY Relative Strength',
                   font=dict(size=16)),
        yaxis_title=f'{etf_name}/SPY Ratio',
        height=400,
        **LAYOUT_DEFAULTS,
    )
    return fig


def backtest_summary_chart(results_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing win rates across indicators and horizons."""
    if results_df.empty:
        return go.Figure()

    fig = go.Figure()

    colors = {'3m': YELLOW, '6m': BLUE, '12m': GREEN}
    for horizon, color in colors.items():
        col = f'{horizon}_wr'
        if col in results_df.columns:
            fig.add_trace(go.Bar(
                x=results_df['signal'],
                y=results_df[col] * 100,
                name=f'{horizon} Win Rate',
                marker_color=color,
                text=results_df[col].apply(lambda x: f'{x*100:.0f}%' if pd.notna(x) else ''),
                textposition='outside',
                textfont=dict(size=9),
            ))

    fig.update_layout(
        title=dict(text='Backtest Win Rates by Indicator', font=dict(size=16)),
        yaxis_title='Win Rate (%)',
        barmode='group',
        height=500,
        xaxis_tickangle=-45,
        **LAYOUT_DEFAULTS,
    )
    fig.add_hline(y=50, line_dash='dot', line_color=GRAY,
                  annotation_text='50% (coin flip)', annotation_font_color=GRAY)
    return fig


def indicator_spy_chart(indicator_df: pd.DataFrame, spy_df: pd.DataFrame,
                        date_col: str, value_col: str,
                        title: str = 'Indicator vs SPY',
                        y_label: str = 'Indicator',
                        threshold: float = None,
                        threshold_label: str = None,
                        threshold_direction: str = 'above',
                        invert_y: bool = False,
                        value_col_2: str = None,
                        label_1: str = None,
                        label_2: str = None,
                        color_1: str = YELLOW,
                        color_2: str = '#bc8cff',
                        height: int = 350,
                        spy_label: str = 'SPY') -> go.Figure:
    """Dual-axis chart: indicator on left y-axis, asset price on right y-axis.

    Args:
        indicator_df: DataFrame with date and value column(s)
        spy_df: DataFrame with 'date' and 'close' columns
        date_col: column name for dates in indicator_df
        value_col: primary indicator column
        threshold: optional horizontal threshold line
        threshold_direction: 'above' or 'below' — which side is the signal zone
        invert_y: if True, reverse the y-axis (lower = higher on chart)
        value_col_2: optional second indicator line (e.g., 30d MA alongside 10d MA)
        label_1, label_2: legend labels for the indicator lines
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary indicator line
    name_1 = label_1 or value_col
    fig.add_trace(go.Scatter(
        x=indicator_df[date_col], y=indicator_df[value_col],
        mode='lines', name=name_1,
        line=dict(color=color_1, width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>' + name_1 + ': %{y:.2f}<extra></extra>',
    ), secondary_y=False)

    # Optional second indicator line
    if value_col_2 and value_col_2 in indicator_df.columns:
        name_2 = label_2 or value_col_2
        fig.add_trace(go.Scatter(
            x=indicator_df[date_col], y=indicator_df[value_col_2],
            mode='lines', name=name_2,
            line=dict(color=color_2, width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>' + name_2 + ': %{y:.2f}<extra></extra>',
        ), secondary_y=False)

    # Threshold line + signal zone shading
    if threshold is not None:
        t_label = threshold_label or f'Threshold: {threshold}'
        fig.add_hline(y=threshold, line_dash='dash', line_color=GREEN,
                      annotation_text=t_label,
                      annotation_font_color=GREEN, opacity=0.6,
                      secondary_y=False)

        # Highlight periods where indicator is in signal zone
        vals = indicator_df[value_col].values
        dates = indicator_df[date_col].values
        in_signal = False
        start_date = None
        for i in range(len(vals)):
            v = vals[i]
            if pd.isna(v):
                continue
            triggered = (v > threshold) if threshold_direction == 'above' else (v < threshold)
            if triggered and not in_signal:
                in_signal = True
                start_date = dates[i]
            elif not triggered and in_signal:
                in_signal = False
                fig.add_vrect(
                    x0=pd.Timestamp(start_date), x1=pd.Timestamp(dates[i]),
                    fillcolor=GREEN, opacity=0.08, layer='below', line_width=0)
        # Close last region if still in signal
        if in_signal and start_date is not None:
            fig.add_vrect(
                x0=pd.Timestamp(start_date), x1=pd.Timestamp(dates[-1]),
                fillcolor=GREEN, opacity=0.08, layer='below', line_width=0)

    # SPY on secondary axis
    if spy_df is not None and not spy_df.empty:
        # Filter SPY to indicator date range
        min_date = indicator_df[date_col].min()
        max_date = indicator_df[date_col].max()
        spy_range = spy_df[(spy_df['date'] >= min_date) & (spy_df['date'] <= max_date)]
        fig.add_trace(go.Scatter(
            x=spy_range['date'], y=spy_range['close'],
            mode='lines', name=spy_label,
            line=dict(color=BLUE, width=1.2),
            opacity=0.5,
            hovertemplate='%{x|%Y-%m-%d}<br>' + spy_label + ': $%{y:.0f}<extra></extra>',
        ), secondary_y=True)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        **LAYOUT_DEFAULTS,
    )
    fig.update_yaxes(title_text=y_label, secondary_y=False, gridcolor='#16162a')
    if invert_y:
        fig.update_yaxes(autorange='reversed', secondary_y=False)
    fig.update_yaxes(title_text=f'{spy_label} ($)', secondary_y=True, gridcolor='#16162a')

    return fig


def cot_positioning_chart(cot_df: pd.DataFrame, price_df: pd.DataFrame,
                          contract_label: str, ticker: str,
                          sector_color: str = YELLOW,
                          is_equity: bool = False,
                          height: int = 420,
                          trades: list = None) -> go.Figure:
    """Beginner-friendly COT positioning chart with 0-100% scale.

    trades: optional list of dicts from backtest (entry_date, exit_date,
            entry_price, exit_price, direction, return). If provided,
            these are plotted instead of recalculated signals.

    Shows all 3 trader types as percentiles (0-100), price overlay,
    and colored extreme zones (red/green bands at 5th/95th).
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Show full COT history so all backtest trades are visible on the chart
    cot_df = cot_df.copy()

    # Convert fractional percentiles (0.0-1.0) to percentage (0-100) for readability
    has_spec = 'spec_net_pctile' in cot_df.columns
    has_comm = 'comm_net_pctile' in cot_df.columns
    has_small = 'small_spec_net_pctile' in cot_df.columns

    dates = cot_df['date']

    # ── Extreme zone bands ──
    # Top band: >= 95th percentile (crowd is extremely bullish)
    fig.add_hrect(y0=95, y1=100, fillcolor=RED, opacity=0.08, layer='below',
                  line_width=0, secondary_y=False)
    # Bottom band: <= 5th percentile (crowd is extremely bearish)
    fig.add_hrect(y0=0, y1=5, fillcolor=GREEN, opacity=0.08, layer='below',
                  line_width=0, secondary_y=False)

    # Threshold lines
    fig.add_hline(y=95, line_dash='dot', line_color=RED, opacity=0.4, secondary_y=False)
    fig.add_hline(y=5, line_dash='dot', line_color=GREEN, opacity=0.4, secondary_y=False)
    fig.add_hline(y=50, line_dash='dot', line_color=GRAY, opacity=0.2, secondary_y=False)

    # ── Trader type lines ──
    if has_spec:
        spec_vals = cot_df['spec_net_pctile'] * 100
        fig.add_trace(go.Scatter(
            x=dates, y=spec_vals,
            mode='lines', name='Large Speculators',
            line=dict(color=sector_color, width=2.5),
            hovertemplate='%{x|%Y-%m-%d}<br>Large Specs: %{y:.0f}%<extra></extra>',
        ), secondary_y=False)

    if has_comm:
        comm_vals = cot_df['comm_net_pctile'] * 100
        fig.add_trace(go.Scatter(
            x=dates, y=comm_vals,
            mode='lines', name='Commercials',
            line=dict(color='#8888a0', width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>Commercials: %{y:.0f}%<extra></extra>',
        ), secondary_y=False)

    if has_small:
        small_vals = cot_df['small_spec_net_pctile'] * 100
        fig.add_trace(go.Scatter(
            x=dates, y=small_vals,
            mode='lines', name='Small Speculators',
            line=dict(color='#bc8cff', width=1.5, dash='dot'),
            hovertemplate='%{x|%Y-%m-%d}<br>Small Specs: %{y:.0f}%<extra></extra>',
        ), secondary_y=False)

    # ── Price overlay on right axis ──
    price_label = f'{contract_label} Price'
    if price_df is not None and not price_df.empty:
        min_date = dates.min()
        max_date = dates.max()
        price_range = price_df[(price_df['date'] >= min_date) & (price_df['date'] <= max_date)]
        fig.add_trace(go.Scatter(
            x=price_range['date'], y=price_range['close'],
            mode='lines', name=price_label,
            line=dict(color=BLUE, width=1.5),
            opacity=0.45,
            hovertemplate='%{x|%Y-%m-%d}<br>' + price_label + ': $%{y:.2f}<extra></extra>',
        ), secondary_y=True)

    # ── Signal zone shading (vertical bands where setup fires) ──
    spec_vals_raw = cot_df['spec_net_pctile'].values if has_spec else np.full(len(cot_df), 0.5)
    comm_vals_raw = cot_df['comm_net_pctile'].values if has_comm else np.full(len(cot_df), 0.5)
    small_vals_raw = cot_df['small_spec_net_pctile'].values if has_small else np.full(len(cot_df), 0.5)
    date_vals = cot_df['date'].values

    def _find_signal_zones(date_vals, spec_vals, comm_vals, small_vals, is_eq, lo, hi, require_all_three):
        """Find signal zone start/end pairs for given thresholds."""
        zones_long, zones_short = [], []
        in_long = in_short = False
        long_start = short_start = None
        for i in range(len(date_vals)):
            s, c, sm = spec_vals[i], comm_vals[i], small_vals[i]
            if pd.isna(s): s = 0.5
            if pd.isna(c): c = 0.5
            if pd.isna(sm): sm = 0.5
            if is_eq:
                is_long = c > hi
                is_short = c < lo
            elif require_all_three:
                is_long = s < lo and c > hi and sm < lo
                is_short = s > hi and c < lo and sm > hi
            else:
                # Fallback: at least 2 of 3 types at extremes
                long_count = (s < lo) + (c > hi) + (sm < lo)
                short_count = (s > hi) + (c < lo) + (sm > hi)
                is_long = long_count >= 2
                is_short = short_count >= 2
            # Mutually exclusive — long takes priority if both trigger
            if is_long and is_short:
                is_short = False

            if is_long and not in_long:
                if in_short:  # close any open short zone
                    in_short = False; zones_short.append((short_start, date_vals[i]))
                in_long = True; long_start = date_vals[i]
            elif not is_long and in_long:
                in_long = False; zones_long.append((long_start, date_vals[i]))
            if is_short and not in_short:
                if in_long:  # close any open long zone
                    in_long = False; zones_long.append((long_start, date_vals[i]))
                in_short = True; short_start = date_vals[i]
            elif not is_short and in_short:
                in_short = False; zones_short.append((short_start, date_vals[i]))
        if in_long and long_start is not None:
            zones_long.append((long_start, date_vals[-1]))
        if in_short and short_start is not None:
            zones_short.append((short_start, date_vals[-1]))
        return zones_long, zones_short

    # Try strict method first (Method C: all 3 at 5%/95%)
    zones_long, zones_short = _find_signal_zones(
        date_vals, spec_vals_raw, comm_vals_raw, small_vals_raw,
        is_equity, 0.05, 0.95, require_all_three=True)

    # If no strict signals found, try 2-of-3 at 10%/90%, then commercials-only at 5%/95%
    if not zones_long and not zones_short and not is_equity:
        zones_long, zones_short = _find_signal_zones(
            date_vals, spec_vals_raw, comm_vals_raw, small_vals_raw,
            is_equity, 0.10, 0.90, require_all_three=False)
    if not zones_long and not zones_short and not is_equity:
        # Last resort: use commercials only (same as equity rule)
        zones_long, zones_short = _find_signal_zones(
            date_vals, spec_vals_raw, comm_vals_raw, small_vals_raw,
            True, 0.05, 0.95, require_all_three=True)

    for start, end in zones_long:
        fig.add_vrect(x0=pd.Timestamp(start), x1=pd.Timestamp(end),
                      fillcolor=GREEN, opacity=0.18, layer='below', line_width=0)
    for start, end in zones_short:
        fig.add_vrect(x0=pd.Timestamp(start), x1=pd.Timestamp(end),
                      fillcolor=RED, opacity=0.18, layer='below', line_width=0)

    # ── Entry / Exit markers on price line ──
    # Uses actual backtest trades when provided, otherwise falls back to zone-based calculation
    TIFFANY = '#0abab5'
    PINK = '#ff69b4'

    entry_long_x, entry_long_y = [], []
    entry_short_x, entry_short_y = [], []
    exit_x, exit_y, exit_returns = [], [], []

    if trades:
        # Use actual backtest trade data — guaranteed to match the table
        for t in trades:
            ed = pd.Timestamp(t['entry_date'])
            ep = float(t['entry_price'])
            if t['direction'] == 'long':
                entry_long_x.append(ed)
                entry_long_y.append(ep)
            else:
                entry_short_x.append(ed)
                entry_short_y.append(ep)
            if t.get('exit_date') and t.get('exit_price'):
                exit_x.append(pd.Timestamp(t['exit_date']))
                exit_y.append(float(t['exit_price']))
                exit_returns.append(float(t['return']) * 100)

    elif price_df is not None and not price_df.empty and (zones_long or zones_short):
        # Fallback: recalculate from zones (reversal entry + neutral exit)
        _pdf = price_df.sort_values('date').reset_index(drop=True)

        def _find_reversal_entry(signal_date, direction, window=5):
            ts = pd.Timestamp(signal_date)
            candidates = _pdf[(_pdf['date'] > ts) & (_pdf['date'] <= ts + pd.Timedelta(days=window * 2))].head(window)
            for _, row in candidates.iterrows():
                if direction == 'long' and row['close'] > row.get('open', row['close'] - 1):
                    return row['date'], float(row['close'])
                elif direction == 'short' and row['close'] < row.get('open', row['close'] + 1):
                    return row['date'], float(row['close'])
            return None, None

        def _find_neutral_exit(entry_date, max_weeks=52):
            future = cot_df[cot_df['date'] > pd.Timestamp(entry_date)].head(max_weeks)
            for _, row in future.iterrows():
                for col in ['spec_net_pctile', 'comm_net_pctile', 'small_spec_net_pctile']:
                    if col in row.index and not pd.isna(row[col]):
                        if 0.40 <= row[col] <= 0.60:
                            return row['date']
            return None

        def _price_at(dt):
            ts = pd.Timestamp(dt)
            exact = _pdf[_pdf['date'] == ts]
            if not exact.empty:
                return float(exact.iloc[0]['close'])
            near = _pdf[(abs(_pdf['date'] - ts) <= pd.Timedelta(days=7))]
            if not near.empty:
                nearest_idx = (abs(near['date'] - ts)).idxmin()
                return float(near.loc[nearest_idx, 'close'])
            return None

        for start, end in zones_long:
            entry_date, entry_price = _find_reversal_entry(start, 'long')
            if entry_date is None:
                continue
            entry_long_x.append(entry_date)
            entry_long_y.append(entry_price)
            exit_date = _find_neutral_exit(entry_date)
            if exit_date is not None:
                exit_price = _price_at(exit_date)
                if exit_price is not None:
                    exit_x.append(exit_date)
                    exit_y.append(exit_price)
                    exit_returns.append((exit_price - entry_price) / entry_price * 100)

        for start, end in zones_short:
            entry_date, entry_price = _find_reversal_entry(start, 'short')
            if entry_date is None:
                continue
            entry_short_x.append(entry_date)
            entry_short_y.append(entry_price)
            exit_date = _find_neutral_exit(entry_date)
            if exit_date is not None:
                exit_price = _price_at(exit_date)
                if exit_price is not None:
                    exit_x.append(exit_date)
                    exit_y.append(exit_price)
                    exit_returns.append((entry_price - exit_price) / entry_price * 100)

    if entry_long_x or entry_short_x:

        # Long entries — triangle-up
        if entry_long_x:
            fig.add_trace(go.Scatter(
                x=entry_long_x, y=entry_long_y,
                mode='markers', name='Long Entry',
                marker=dict(symbol='triangle-up', size=10, color=TIFFANY,
                            line=dict(width=1, color='white')),
                hovertemplate='%{x|%Y-%m-%d}<br>LONG Entry: $%{y:,.2f}<extra></extra>',
            ), secondary_y=True)

        # Short entries — triangle-down (red)
        if entry_short_x:
            fig.add_trace(go.Scatter(
                x=entry_short_x, y=entry_short_y,
                mode='markers', name='Short Entry',
                marker=dict(symbol='triangle-down', size=10, color=RED,
                            line=dict(width=1, color='white')),
                hovertemplate='%{x|%Y-%m-%d}<br>SHORT Entry: $%{y:,.2f}<extra></extra>',
            ), secondary_y=True)

        # Exit markers — x with return % label (hide labels if too many)
        if exit_x:
            exit_text = [f'{r:+.1f}%' for r in exit_returns]
            exit_colors = [GREEN if r >= 0 else RED for r in exit_returns]
            show_labels = len(exit_x) <= 15
            hover_tpl = [
                f'%{{x|%Y-%m-%d}}<br>Exit: $%{{y:,.2f}}<br>Return: {t}<extra></extra>'
                for t in exit_text
            ]
            fig.add_trace(go.Scatter(
                x=exit_x, y=exit_y,
                mode='markers+text' if show_labels else 'markers',
                name='Exit',
                marker=dict(symbol='x', size=9, color=PINK,
                            line=dict(width=2, color=PINK)),
                text=exit_text if show_labels else None,
                textposition='top center' if show_labels else None,
                textfont=dict(size=9, color=exit_colors) if show_labels else None,
                hovertemplate=hover_tpl,
            ), secondary_y=True)

    # ── Annotations for extreme zones ──
    fig.add_annotation(
        x=0.01, y=98, xref='paper', yref='y',
        text='CROWDED LONG', font=dict(color=RED, size=9),
        showarrow=False, opacity=0.6)
    fig.add_annotation(
        x=0.01, y=2, xref='paper', yref='y',
        text='CROWDED SHORT', font=dict(color=GREEN, size=9),
        showarrow=False, opacity=0.6)

    # ── Layout ──
    equity_note = '<br><span style="font-size:11px;color:#8888a0">Equities: only Commercials needed for setup</span>' if is_equity else ''
    fig.update_layout(
        title=dict(
            text=f'{contract_label} — COT Positioning{equity_note}',
            font=dict(size=14),
            y=0.97, x=0.01, xanchor='left', yanchor='top',
        ),
        height=height,
        **LAYOUT_DEFAULTS,
    )
    # Override legend to horizontal above chart
    fig.update_layout(
        legend=dict(
            bgcolor='rgba(0,0,0,0)', font=dict(size=10),
            orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
        ),
        margin=dict(l=60, r=60, t=50, b=40),
    )

    # Left axis: 0-100% scale, intuitive direction
    fig.update_yaxes(
        title_text='COT Percentile (%)',
        range=[0, 100],
        tickvals=[0, 5, 25, 50, 75, 95, 100],
        ticktext=['0%', '5%', '25%', '50%', '75%', '95%', '100%'],
        secondary_y=False,
        gridcolor='#16162a',
    )

    # Right axis: price
    fig.update_yaxes(
        title_text=f'{contract_label} ($)',
        secondary_y=True,
        gridcolor='#16162a',
    )

    return fig


def analog_chart(analogs_df: pd.DataFrame, indicator_col: str,
                 current_value: float) -> go.Figure:
    """Show historical analogs — when the indicator was at a similar level and SPY forward returns."""
    if analogs_df.empty:
        return go.Figure()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Historical Dates with Similar Reading',
                                        'Forward Returns Distribution'],
                        column_widths=[0.5, 0.5])

    # Left: scatter of analog dates vs indicator value
    fig.add_trace(go.Scatter(
        x=analogs_df['date'],
        y=analogs_df[indicator_col],
        mode='markers+text',
        marker=dict(color=BLUE, size=10),
        text=analogs_df['date'].dt.strftime('%Y-%m'),
        textposition='top center',
        textfont=dict(size=8, color=TEXT),
        name='Analog dates',
    ), row=1, col=1)
    fig.add_hline(y=current_value, line_dash='dash', line_color=YELLOW,
                  annotation_text=f'Current: {current_value:.3f}',
                  annotation_font_color=YELLOW, row=1, col=1)

    # Right: forward return bars
    for horizon, color in [('3m', YELLOW), ('6m', BLUE), ('12m', GREEN)]:
        col_name = f'fwd_{horizon}'
        if col_name in analogs_df.columns:
            vals = analogs_df[col_name].dropna() * 100
            if not vals.empty:
                fig.add_trace(go.Box(
                    y=vals, name=horizon,
                    marker_color=color,
                    boxmean=True,
                ), row=1, col=2)

    fig.update_layout(
        title=dict(text=f'Historical Indicator Analogs', font=dict(size=16)),
        height=400,
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )
    return fig


def signal_timeline_chart(signal_history: dict,
                          title: str = 'Signal History — Last 5 Years') -> go.Figure:
    """Horizontal timeline: one row per indicator, dots colored by forward 3m return.

    Args:
        signal_history: {indicator_name: [(pd.Timestamp, fwd_3m_return), ...]}
    """
    fig = go.Figure()
    indicators = list(signal_history.keys())

    for name, signals in signal_history.items():
        if not signals:
            continue
        dates = [s[0] for s in signals]
        returns = [s[1] for s in signals]
        colors = []
        hover_texts = []
        for r in returns:
            if r is None or pd.isna(r):
                colors.append(GRAY)
                hover_texts.append("pending")
            elif r > 0:
                colors.append(GREEN)
                hover_texts.append(f"+{r*100:.1f}%")
            else:
                colors.append(RED)
                hover_texts.append(f"{r*100:.1f}%")

        fig.add_trace(go.Scatter(
            x=dates, y=[name] * len(dates),
            mode='markers',
            marker=dict(color=colors, size=9,
                        line=dict(color='rgba(255,255,255,0.3)', width=0.5)),
            hovertemplate='%{x|%Y-%m-%d}<br>3m return: %{customdata}<extra></extra>',
            customdata=hover_texts,
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=max(280, len(indicators) * 35 + 100),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family='Inter, sans-serif', size=12),
        margin=dict(l=10, r=10, t=40, b=30),
        hovermode='x unified',
    )
    fig.update_xaxes(gridcolor='#16162a')
    fig.update_yaxes(gridcolor='#16162a', categoryorder='array', categoryarray=indicators[::-1])

    return fig


# ══════════════════════════════════════════════════════════════════════════
# New experimental charts
# ══════════════════════════════════════════════════════════════════════════

def dma50_zscore_chart(price_df: pd.DataFrame, label: str = 'S&P 500',
                       height: int = 380) -> go.Figure:
    """Distance from 50-DMA as a Z-score. Works for any asset (SPY, BTC, etc).

    Top panel: Price with 50-DMA line
    Bottom panel: Z-score oscillator with overbought/oversold bands
    """
    df = price_df.copy().sort_values('date').reset_index(drop=True)
    df['ma50'] = df['close'].rolling(50).mean()
    df['spread'] = (df['close'] / df['ma50']) - 1
    df['spread_std'] = df['spread'].rolling(252).std()
    df['zscore'] = df['spread'] / df['spread_std']
    df = df.dropna(subset=['zscore'])

    cutoff = df['date'].max() - pd.DateOffset(years=2)
    df = df[df['date'] >= cutoff]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.55], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'], mode='lines', name=label,
        line=dict(color=BLUE, width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>' + label + ': $%{y:,.2f}<extra></extra>',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['ma50'], mode='lines', name='50-DMA',
        line=dict(color=YELLOW, width=1.5, dash='dot'),
        hovertemplate='%{x|%Y-%m-%d}<br>50-DMA: $%{y:,.2f}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['zscore'], mode='lines', name='Z-Score',
        line=dict(color=TEXT, width=1.5),
        hovertemplate='%{x|%Y-%m-%d}<br>Z-Score: %{y:.2f}<extra></extra>',
    ), row=2, col=1)

    fig.add_hrect(y0=2, y1=4, fillcolor=RED, opacity=0.12, layer='below',
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=1, y1=2, fillcolor=RED, opacity=0.06, layer='below',
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=-2, y1=-1, fillcolor=GREEN, opacity=0.06, layer='below',
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=-4, y1=-2, fillcolor=GREEN, opacity=0.12, layer='below',
                  line_width=0, row=2, col=1)

    for y_val, clr in [(2, RED), (1, RED), (-1, GREEN), (-2, GREEN), (0, GRAY)]:
        fig.add_hline(y=y_val, line_dash='dot', line_color=clr, opacity=0.4, row=2, col=1)

    fig.add_annotation(x=0.01, y=2.5, xref='paper', yref='y2',
                       text='Extreme Overbought', font=dict(color=RED, size=9),
                       showarrow=False, opacity=0.6)
    fig.add_annotation(x=0.01, y=-2.5, xref='paper', yref='y2',
                       text='Extreme Oversold', font=dict(color=GREEN, size=9),
                       showarrow=False, opacity=0.6)

    curr_z = df['zscore'].iloc[-1]
    z_color = RED if curr_z > 1 else GREEN if curr_z < -1 else TEXT
    fig.add_annotation(
        x=df['date'].iloc[-1], y=curr_z, xref='x2', yref='y2',
        text=f'  {curr_z:.1f}', font=dict(color=z_color, size=12, family='monospace'),
        showarrow=False, xanchor='left')

    fig.update_layout(
        title=dict(text=f'{label} — 50-DMA Spread Z-Score', font=dict(size=14),
                   y=0.98, x=0.01, xanchor='left', yanchor='top'),
        height=height,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family='Inter, sans-serif', size=12),
        margin=dict(l=60, r=30, t=40, b=30),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_xaxes(gridcolor='#16162a')
    fig.update_yaxes(gridcolor='#16162a', row=1, col=1)
    fig.update_yaxes(gridcolor='#16162a', title_text='Z-Score (σ)', row=2, col=1)

    return fig


def technical_composite_chart(price_df: pd.DataFrame, label: str = 'S&P 500',
                              height: int = 380) -> go.Figure:
    """Technical Composite oscillator. Works for any asset (SPY, BTC, etc).

    Combines 4 components: 50-DMA z-score, RSI(14), Bollinger Band, MACD histogram.
    """
    df = price_df.copy().sort_values('date').reset_index(drop=True)

    df['ma50'] = df['close'].rolling(50).mean()
    spread = (df['close'] / df['ma50']) - 1
    df['c1_dma'] = spread / spread.rolling(252).std()

    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df['c2_rsi'] = (rsi - 50) / 25

    ma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['c3_bband'] = (df['close'] - ma20) / std20

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = macd_line - signal_line
    df['c4_macd'] = macd_hist / macd_hist.rolling(252).std()

    df['composite'] = df[['c1_dma', 'c2_rsi', 'c3_bband', 'c4_macd']].mean(axis=1)
    df = df.dropna(subset=['composite'])

    cutoff = df['date'].max() - pd.DateOffset(years=2)
    df = df[df['date'] >= cutoff]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.40, 0.60], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'], mode='lines', name=label,
        line=dict(color=BLUE, width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>' + label + ': $%{y:,.2f}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['composite'], mode='lines', name='Technical Composite',
        line=dict(color=TEXT, width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Composite: %{y:.2f}<extra></extra>',
    ), row=2, col=1)

    fig.add_hrect(y0=1.5, y1=4, fillcolor=RED, opacity=0.10, layer='below',
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=-4, y1=-1.5, fillcolor=GREEN, opacity=0.10, layer='below',
                  line_width=0, row=2, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color=GRAY, opacity=0.3, row=2, col=1)
    fig.add_hline(y=1.5, line_dash='dot', line_color=RED, opacity=0.4, row=2, col=1)
    fig.add_hline(y=-1.5, line_dash='dot', line_color=GREEN, opacity=0.4, row=2, col=1)

    fig.add_annotation(x=0.01, y=2.5, xref='paper', yref='y2',
                       text='Overbought (Selloff Likely)', font=dict(color=RED, size=9),
                       showarrow=False, opacity=0.6)
    fig.add_annotation(x=0.01, y=-2.5, xref='paper', yref='y2',
                       text='Oversold (Bounce Likely)', font=dict(color=GREEN, size=9),
                       showarrow=False, opacity=0.6)

    curr = df['composite'].iloc[-1]
    c_color = RED if curr > 1 else GREEN if curr < -1 else TEXT
    fig.add_annotation(
        x=df['date'].iloc[-1], y=curr, xref='x2', yref='y2',
        text=f'  {curr:.1f}', font=dict(color=c_color, size=12, family='monospace'),
        showarrow=False, xanchor='left')

    fig.update_layout(
        title=dict(text=f'{label} — Technical Composite (RSI + MACD + BBand + 50-DMA)',
                   font=dict(size=14), y=0.98, x=0.01, xanchor='left', yanchor='top'),
        height=height,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family='Inter, sans-serif', size=12),
        margin=dict(l=60, r=30, t=40, b=30),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_xaxes(gridcolor='#16162a')
    fig.update_yaxes(gridcolor='#16162a', row=1, col=1)
    fig.update_yaxes(gridcolor='#16162a', title_text='Score', row=2, col=1)

    return fig


# NOTE: asset_manager_positioning_chart removed — TFF data didn't provide actionable alpha.


def _deprecated_asset_manager_positioning_chart(price_df: pd.DataFrame,
                                     height: int = 420) -> go.Figure:
    """Chart 2: Asset Manager Net Position weekly change (from CFTC TFF report).

    Uses real Traders in Financial Futures data.
    - Top: SPY price with green/red background shading at extreme weeks
    - Bottom: Weekly change as a line with colored fill to zero
    """
    from pathlib import Path
    tff_path = Path(__file__).parent / 'data' / 'tff_sp500_asset_mgr.csv'
    if not tff_path.exists():
        return go.Figure()

    df = pd.read_csv(tff_path, parse_dates=['date']).sort_values('date').reset_index(drop=True)

    # Rolling z-score for extreme detection
    df['change_mean'] = df['am_net_change'].rolling(104).mean()
    df['change_std'] = df['am_net_change'].rolling(104).std()
    df['change_z'] = (df['am_net_change'] - df['change_mean']) / df['change_std']
    df = df.dropna(subset=['change_z'])

    # Last 5 years
    cutoff = df['date'].max() - pd.DateOffset(years=5)
    df = df[df['date'] >= cutoff].reset_index(drop=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.40, 0.60], vertical_spacing=0.04)

    # ── Top: SPY price ──
    if price_df is not None and not price_df.empty:
        p = price_df[price_df['date'] >= cutoff]
        fig.add_trace(go.Scatter(
            x=p['date'], y=p['close'], mode='lines', name='S&P 500',
            line=dict(color=BLUE, width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>SPY: $%{y:.2f}<extra></extra>',
        ), row=1, col=1)

    # Add green/red vrects on price panel for extreme weeks
    for i, row in df.iterrows():
        z = row['change_z']
        if z <= -2:
            fig.add_vrect(x0=row['date'] - pd.Timedelta(days=3),
                          x1=row['date'] + pd.Timedelta(days=3),
                          fillcolor=RED, opacity=0.15, layer='below', line_width=0,
                          row=1, col=1)
        elif z >= 2:
            fig.add_vrect(x0=row['date'] - pd.Timedelta(days=3),
                          x1=row['date'] + pd.Timedelta(days=3),
                          fillcolor=GREEN, opacity=0.15, layer='below', line_width=0,
                          row=1, col=1)

    # ── Bottom: Weekly change as line with fill ──
    # Positive fill (green)
    pos_y = [max(0, v) for v in df['am_net_change']]
    neg_y = [min(0, v) for v in df['am_net_change']]

    fig.add_trace(go.Scatter(
        x=df['date'], y=pos_y, mode='lines', name='Buying',
        line=dict(color=GREEN, width=0.5),
        fill='tozeroy', fillcolor='rgba(63,185,80,0.3)',
        hoverinfo='skip', showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df['date'], y=neg_y, mode='lines', name='Selling',
        line=dict(color=RED, width=0.5),
        fill='tozeroy', fillcolor='rgba(255,107,107,0.3)',
        hoverinfo='skip', showlegend=False,
    ), row=2, col=1)

    # Main line on top
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['am_net_change'], mode='lines',
        name='Asset Mgr Δ Net',
        line=dict(color=TEXT, width=1),
        hovertemplate='%{x|%Y-%m-%d}<br>Δ Net: %{y:,.0f}<br>z=%{customdata:.1f}<extra></extra>',
        customdata=df['change_z'],
    ), row=2, col=1)

    # Zero line
    fig.add_hline(y=0, line_dash='solid', line_color=TEXT, opacity=0.3, row=2, col=1)

    # Current value
    curr_z = df['change_z'].iloc[-1]
    curr_chg = df['am_net_change'].iloc[-1]
    z_color = RED if curr_z < -1 else GREEN if curr_z > 1 else TEXT
    fig.add_annotation(
        x=df['date'].iloc[-1], y=curr_chg, xref='x2', yref='y2',
        text=f'  z={curr_z:.1f}', font=dict(color=z_color, size=12, family='monospace'),
        showarrow=False, xanchor='left')

    fig.update_layout(
        title=dict(text="S&P 500 — Asset Manager Net Position Weekly Change (TFF)",
                   font=dict(size=14), y=0.98, x=0.01, xanchor='left', yanchor='top'),
        height=height,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT, family='Inter, sans-serif', size=12),
        margin=dict(l=60, r=30, t=40, b=30),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_xaxes(gridcolor='#16162a')
    fig.update_yaxes(gridcolor='#16162a', row=1, col=1)
    fig.update_yaxes(gridcolor='#16162a', title_text='Δ Net Position', row=2, col=1)

    return fig
