# monthly_ftse_momentum_rebalance.py

import os
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from datetime import datetime

# ======================================================
# SETTINGS
# ======================================================

TOP_N = 10

PRICE_SCALE = 0.01

MIN_PRICE_GBP = 0.50
MIN_AVG_VALUE_TRADED_GBP = 1_000_000

EXCLUDED_SECTORS = ["investment trusts"]

LOOKBACK_LONG_DAYS = 84
LOOKBACK_SHORT_DAYS = 42
SKIP_DAYS = 21

WEIGHT_SHORT = 0.50
WEIGHT_LONG = 0.50

MAX_ALLOWED_MOMENTUM = 2.5
MAX_SINGLE_DAY_MOVE = 0.50
MAX_ANNUALISED_VOL = 1.80

MIN_VALID_DAYS = max(LOOKBACK_LONG_DAYS, LOOKBACK_SHORT_DAYS, SKIP_DAYS)

MARKET_PROXY = "^FTLC"

TIME_STOP_DAYS = 56

PORTFOLIO_FILE = "portfolio_ftse.csv"
HTML_FILE = "ftse_monthly_email.html"


# ======================================================
# GET FTSE TABLES
# ======================================================

def get_ftse_table_from_wikipedia(url, source_index_name):
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    for table in tables:
        cols = [str(c).strip().lower() for c in table.columns]
        joined = " ".join(cols)

        if "company" in joined and "ticker" in joined:
            table = table.copy()
            table.columns = [str(c).strip() for c in table.columns]

            company_col = None
            ticker_col = None
            sector_col = None

            for col in table.columns:
                col_lower = col.lower()

                if "company" in col_lower:
                    company_col = col

                if "ticker" in col_lower:
                    ticker_col = col

                if "sector" in col_lower:
                    sector_col = col

            if company_col is None or ticker_col is None:
                continue

            if sector_col is None:
                table["Sector"] = "Unknown"
                sector_col = "Sector"

            output = table[[ticker_col, company_col, sector_col]].copy()
            output.columns = ["TickerRaw", "Name", "Sector"]

            output["TickerRaw"] = output["TickerRaw"].astype(str).str.strip()
            output["Name"] = output["Name"].astype(str).str.strip()
            output["Sector"] = output["Sector"].astype(str).str.strip()

            output = output[output["TickerRaw"] != ""]
            output = output.dropna(subset=["TickerRaw", "Name"])

            output["Ticker"] = (
                output["TickerRaw"]
                .str.replace(".", "-", regex=False)
                + ".L"
            )

            output["Index"] = source_index_name

            return output[["Ticker", "Name", "Sector", "Index"]].drop_duplicates()

    raise ValueError(f"Could not find constituents table for {source_index_name}")


def get_ftse350_table():
    ftse100_url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    ftse250_url = "https://en.wikipedia.org/wiki/FTSE_250_Index"

    ftse100 = get_ftse_table_from_wikipedia(ftse100_url, "FTSE 100")
    ftse250 = get_ftse_table_from_wikipedia(ftse250_url, "FTSE 250")

    ftse350 = pd.concat([ftse100, ftse250], ignore_index=True)
    ftse350 = ftse350.drop_duplicates(subset=["Ticker"])

    ftse350 = ftse350[
        ~ftse350["Sector"].str.lower().isin(EXCLUDED_SECTORS)
    ]

    print(f"FTSE 100 tickers: {len(ftse100)}")
    print(f"FTSE 250 tickers: {len(ftse250)}")
    print(f"Combined FTSE 350 tickers after exclusions: {len(ftse350)}")

    return ftse350[["Ticker", "Name", "Sector"]]


# ======================================================
# DOWNLOAD DATA
# ======================================================

def download_data(tickers):
    print(f"Downloading {len(tickers)} tickers...")

    return yf.download(
        tickers=tickers,
        period="1y",
        auto_adjust=True,
        group_by="ticker",
        progress=True,
        threads=True,
    )


# ======================================================
# SAFE DATA EXTRACTION
# ======================================================

def get_ticker_data(data, ticker):
    try:
        df = data[ticker].copy()
        df = df.dropna()
        return df
    except Exception:
        return None


# ======================================================
# 2/4 BLENDED MOMENTUM
# ======================================================

def calculate_momentum_components(close):
    close = close.dropna()

    if len(close) < MIN_VALID_DAYS:
        return None

    try:
        price_now = close.iloc[-1]
        price_short = close.iloc[-LOOKBACK_SHORT_DAYS]
        price_long = close.iloc[-LOOKBACK_LONG_DAYS]
        price_skip = close.iloc[-SKIP_DAYS]

        if min(price_now, price_short, price_long, price_skip) <= 0:
            return None

        momentum_2m = price_now / price_short - 1
        momentum_4m_ex_1m = price_skip / price_long - 1

        blended_momentum = (
            WEIGHT_SHORT * momentum_2m
            + WEIGHT_LONG * momentum_4m_ex_1m
        )

        return {
            "Momentum2M": momentum_2m,
            "Momentum4MEx1M": momentum_4m_ex_1m,
            "BlendedMomentum": blended_momentum,
        }

    except Exception:
        return None


# ======================================================
# DATA QUALITY FILTERS
# ======================================================

def passes_data_quality_filters(close):
    close = close.dropna()

    if len(close) < MIN_VALID_DAYS:
        return False, "Insufficient history"

    daily_returns = close.pct_change().dropna()

    if daily_returns.empty:
        return False, "No return data"

    max_daily_move = daily_returns.abs().max()

    if max_daily_move > MAX_SINGLE_DAY_MOVE:
        return False, f"Extreme daily move: {max_daily_move:.2%}"

    annualised_vol = daily_returns.tail(252).std() * (252 ** 0.5)

    if annualised_vol > MAX_ANNUALISED_VOL:
        return False, f"Extreme volatility: {annualised_vol:.2%}"

    components = calculate_momentum_components(close)

    if components is None:
        return False, "Momentum unavailable"

    blended_momentum = components["BlendedMomentum"]

    if blended_momentum <= 0:
        return False, "Negative blended momentum"

    if blended_momentum > MAX_ALLOWED_MOMENTUM:
        return False, f"Extreme blended momentum: {blended_momentum:.2%}"

    return True, "OK"


# ======================================================
# PORTFOLIO CSV
# ======================================================

def load_current_portfolio():
    required_columns = ["Ticker", "Name", "EntryDate", "EntryPrice"]

    if not os.path.exists(PORTFOLIO_FILE):
        return pd.DataFrame(columns=required_columns)

    df = pd.read_csv(PORTFOLIO_FILE)

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["EntryDate"] = df["EntryDate"].astype(str).str.strip()
    df["EntryPrice"] = pd.to_numeric(df["EntryPrice"], errors="coerce")

    return df[required_columns]


def save_new_portfolio(new_portfolio):
    new_portfolio[["Ticker", "Name", "EntryDate", "EntryPrice"]].to_csv(
        PORTFOLIO_FILE,
        index=False,
    )


# ======================================================
# TOP PORTFOLIO
# ======================================================

def calculate_ranked_universe(data, ftse350_table):
    tickers = ftse350_table["Ticker"].tolist()
    sector_map = dict(zip(ftse350_table["Ticker"], ftse350_table["Sector"]))
    name_map = dict(zip(ftse350_table["Ticker"], ftse350_table["Name"]))

    results = []
    rejected_count = 0

    for ticker in tickers:

        sector = sector_map.get(ticker, "Unknown")
        name = name_map.get(ticker, "Unknown")

        if sector.lower() in EXCLUDED_SECTORS:
            rejected_count += 1
            continue

        df = get_ticker_data(data, ticker)

        if df is None or df.empty:
            rejected_count += 1
            continue

        if "Close" not in df.columns or "Volume" not in df.columns:
            rejected_count += 1
            continue

        close = df["Close"].dropna()
        volume = df["Volume"].dropna()

        if len(close) < MIN_VALID_DAYS or len(volume) < MIN_VALID_DAYS:
            rejected_count += 1
            continue

        latest_price_raw = close.iloc[-1]
        latest_price_gbp = latest_price_raw * PRICE_SCALE

        avg_value_traded_gbp = (
            close * PRICE_SCALE * volume
        ).rolling(20).mean().iloc[-1]

        if latest_price_gbp < MIN_PRICE_GBP:
            rejected_count += 1
            continue

        if (
            pd.isna(avg_value_traded_gbp)
            or avg_value_traded_gbp < MIN_AVG_VALUE_TRADED_GBP
        ):
            rejected_count += 1
            continue

        quality_ok, _ = passes_data_quality_filters(close)

        if not quality_ok:
            rejected_count += 1
            continue

        components = calculate_momentum_components(close)

        if components is None:
            rejected_count += 1
            continue

        results.append({
            "Ticker": ticker,
            "Name": name,
            "Sector": sector,
            "PriceRaw": latest_price_raw,
            "PriceGBP": latest_price_gbp,
            "Momentum2M": components["Momentum2M"],
            "Momentum4MEx1M": components["Momentum4MEx1M"],
            "BlendedMomentum": components["BlendedMomentum"],
        })

    ranked = pd.DataFrame(results)

    if ranked.empty:
        return ranked, rejected_count

    ranked = ranked.sort_values("BlendedMomentum", ascending=False).reset_index(drop=True)
    ranked["RawRank"] = ranked.index + 1

    return ranked, rejected_count


# ======================================================
# TIME STOP AND SELECTION
# ======================================================

def build_new_portfolio(ranked, current_portfolio, data, run_date):
    current_map = {
        row["Ticker"]: row
        for _, row in current_portfolio.iterrows()
    }

    excluded_for_time_stop = set()
    time_stop_rows = []

    for ticker, row in current_map.items():
        df = get_ticker_data(data, ticker)

        if df is None or df.empty or "Close" not in df.columns:
            continue

        close = df["Close"].dropna()

        if close.empty:
            continue

        current_price_raw = close.iloc[-1]
        entry_price = row["EntryPrice"]

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        try:
            entry_date = pd.to_datetime(row["EntryDate"])
        except Exception:
            continue

        holding_days = (pd.to_datetime(run_date) - entry_date).days
        return_since_entry = current_price_raw / entry_price - 1

        if holding_days >= TIME_STOP_DAYS and return_since_entry <= 0:
            excluded_for_time_stop.add(ticker)

            time_stop_rows.append({
                "Ticker": ticker,
                "Name": row["Name"],
                "HoldingDays": holding_days,
                "ReturnSinceEntry": return_since_entry,
            })

    eligible = ranked[~ranked["Ticker"].isin(excluded_for_time_stop)].copy()
    selected = eligible.head(TOP_N).copy()

    selected = selected.reset_index(drop=True)
    selected["PortfolioRank"] = selected.index + 1

    new_rows = []

    for _, row in selected.iterrows():
        ticker = row["Ticker"]

        if ticker in current_map:
            entry_date = current_map[ticker]["EntryDate"]
            entry_price = current_map[ticker]["EntryPrice"]
        else:
            entry_date = run_date
            entry_price = row["PriceRaw"]

        new_rows.append({
            "Ticker": ticker,
            "Name": row["Name"],
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "PortfolioRank": row["PortfolioRank"],
            "RawRank": row["RawRank"],
            "Sector": row["Sector"],
            "Momentum2M": row["Momentum2M"],
            "Momentum4MEx1M": row["Momentum4MEx1M"],
            "BlendedMomentum": row["BlendedMomentum"],
        })

    new_portfolio = pd.DataFrame(new_rows)

    return new_portfolio, pd.DataFrame(time_stop_rows), excluded_for_time_stop


def build_actions(current_portfolio, new_portfolio, time_stop_df):
    current_tickers = set(current_portfolio["Ticker"].tolist())
    new_tickers = set(new_portfolio["Ticker"].tolist()) if not new_portfolio.empty else set()

    sells = sorted(list(current_tickers - new_tickers))
    buys = sorted(list(new_tickers - current_tickers))
    holds = sorted(list(current_tickers & new_tickers))

    time_stop_tickers = set(time_stop_df["Ticker"].tolist()) if not time_stop_df.empty else set()

    return sells, buys, holds, time_stop_tickers


# ======================================================
# HTML EMAIL
# ======================================================

def build_html_email(
    run_date,
    market_momentum,
    mode,
    ranked,
    new_portfolio,
    current_portfolio,
    sells,
    buys,
    holds,
    time_stop_tickers,
    time_stop_df,
):
    if mode == "RISK-OFF":
        header_colour = "#b00020"
        action = "ACTION REQUIRED: CLOSE ALL FTSE POSITIONS"
        summary = "FTSE 350 2/4 blended momentum is negative or zero. The FTSE system is risk-off."
    else:
        header_colour = "#0b6b3a"
        action = "MONTHLY FTSE REBALANCE REQUIRED"
        summary = "FTSE 350 2/4 blended momentum remains positive. Rebalance to the new top 10 portfolio."

    current_name_map = dict(zip(current_portfolio["Ticker"], current_portfolio["Name"])) if not current_portfolio.empty else {}
    new_name_map = dict(zip(new_portfolio["Ticker"], new_portfolio["Name"])) if not new_portfolio.empty else {}

    action_rows = ""

    for ticker in sells:
        reason = "SELL - TIME STOP" if ticker in time_stop_tickers else "SELL"
        colour = "#b00020"
        name = current_name_map.get(ticker, "")
        action_rows += f"""
        <tr>
            <td style="color:{colour};"><strong>{reason}</strong></td>
            <td><strong>{ticker}</strong></td>
            <td>{name}</td>
        </tr>
        """

    for ticker in buys:
        name = new_name_map.get(ticker, "")
        action_rows += f"""
        <tr>
            <td style="color:#0b6b3a;"><strong>BUY</strong></td>
            <td><strong>{ticker}</strong></td>
            <td>{name}</td>
        </tr>
        """

    for ticker in holds:
        name = new_name_map.get(ticker, current_name_map.get(ticker, ""))
        action_rows += f"""
        <tr>
            <td style="color:#555555;"><strong>HOLD</strong></td>
            <td><strong>{ticker}</strong></td>
            <td>{name}</td>
        </tr>
        """

    if not action_rows:
        action_rows = """
        <tr>
            <td colspan="3"><strong>No actions.</strong></td>
        </tr>
        """

    time_stop_html = ""

    if time_stop_df is not None and not time_stop_df.empty:
        rows = ""
        for _, row in time_stop_df.iterrows():
            rows += f"""
            <tr>
                <td><strong>{row["Ticker"]}</strong></td>
                <td>{row["Name"]}</td>
                <td>{int(row["HoldingDays"])}</td>
                <td>{row["ReturnSinceEntry"]:.2%}</td>
            </tr>
            """

        time_stop_html = f"""
        <h2>8-Week No-Progress Exits</h2>
        <table>
            <tr>
                <th>Ticker</th>
                <th>Name</th>
                <th>Holding Days</th>
                <th>Return Since Entry</th>
            </tr>
            {rows}
        </table>
        """

    portfolio_rows = ""

    if new_portfolio is not None and not new_portfolio.empty:
        for _, row in new_portfolio.iterrows():
            portfolio_rows += f"""
            <tr>
                <td>{int(row["PortfolioRank"])}</td>
                <td>{int(row["RawRank"])}</td>
                <td><strong>{row["Ticker"]}</strong></td>
                <td>{row["Name"]}</td>
                <td>{row["Sector"]}</td>
                <td>{row["Momentum2M"]:.2%}</td>
                <td>{row["Momentum4MEx1M"]:.2%}</td>
                <td><strong>{row["BlendedMomentum"]:.2%}</strong></td>
            </tr>
            """

    portfolio_html = ""

    if portfolio_rows:
        portfolio_html = f"""
        <h2>New FTSE Top 10 Portfolio</h2>
        <table>
            <tr>
                <th>Portfolio Rank</th>
                <th>Raw Rank</th>
                <th>Ticker</th>
                <th>Name</th>
                <th>Sector</th>
                <th>2M Momentum</th>
                <th>4M Ex-1M</th>
                <th>Blended</th>
            </tr>
            {portfolio_rows}
        </table>
        """

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                color: #222222;
                line-height: 1.5;
                background: #ffffff;
            }}
            .container {{
                max-width: 1050px;
                margin: auto;
                padding: 20px;
            }}
            .header {{
                background: {header_colour};
                color: #ffffff;
                padding: 20px;
                border-radius: 8px;
            }}
            .header h1 {{
                margin: 0 0 8px 0;
                font-size: 26px;
            }}
            .header h2 {{
                margin: 0;
                font-size: 20px;
            }}
            .box {{
                background: #f5f5f5;
                padding: 16px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
            }}
            th, td {{
                border: 1px solid #dddddd;
                padding: 8px;
                text-align: left;
                font-size: 14px;
            }}
            th {{
                background: #eeeeee;
            }}
            .footer {{
                margin-top: 25px;
                font-size: 13px;
                color: #666666;
            }}
        </style>
    </head>
    <body>
        <div class="container">

            <div class="header">
                <h1>Monthly FTSE Momentum Rebalance</h1>
                <h2>{action}</h2>
            </div>

            <div class="box">
                <p><strong>Run date:</strong> {run_date}</p>
                <p><strong>Market proxy:</strong> {MARKET_PROXY}</p>
                <p><strong>FTSE 350 blended momentum:</strong> {market_momentum:.2%}</p>
                <p><strong>Mode:</strong> {mode}</p>
                <p>{summary}</p>
            </div>

            <h2>Required Actions</h2>
            <table>
                <tr>
                    <th>Action</th>
                    <th>Ticker</th>
                    <th>Name</th>
                </tr>
                {action_rows}
            </table>

            {time_stop_html}

            {portfolio_html}

            <div class="footer">
                <p>
                    portfolio_ftse.csv has been updated automatically.
                    You only need to make the trades shown above.
                </p>
            </div>

        </div>
    </body>
    </html>
    """


# ======================================================
# MAIN REBALANCE
# ======================================================

def run_monthly_rebalance():
    run_date = datetime.now().strftime("%Y-%m-%d")

    current_portfolio = load_current_portfolio()

    ftse350_table = get_ftse350_table()
    tickers = ftse350_table["Ticker"].tolist()

    all_tickers = sorted(list(set(tickers + [MARKET_PROXY] + current_portfolio["Ticker"].tolist())))

    data = download_data(all_tickers)

    market_df = get_ticker_data(data, MARKET_PROXY)

    if market_df is None or market_df.empty:
        raise ValueError("Could not retrieve FTSE 350 market proxy data.")

    market_components = calculate_momentum_components(market_df["Close"])

    if market_components is None:
        raise ValueError("Could not calculate FTSE 350 blended momentum.")

    market_momentum = market_components["BlendedMomentum"]
    mode = "RISK-ON" if market_momentum > 0 else "RISK-OFF"

    ranked, rejected_count = calculate_ranked_universe(
        data=data,
        ftse350_table=ftse350_table,
    )

    if mode == "RISK-OFF":
        new_portfolio = pd.DataFrame(columns=["Ticker", "Name", "EntryDate", "EntryPrice"])
        time_stop_df = pd.DataFrame()
        sells = sorted(current_portfolio["Ticker"].tolist())
        buys = []
        holds = []
        time_stop_tickers = set()
    else:
        new_portfolio, time_stop_df, _ = build_new_portfolio(
            ranked=ranked,
            current_portfolio=current_portfolio,
            data=data,
            run_date=run_date,
        )

        sells, buys, holds, time_stop_tickers = build_actions(
            current_portfolio=current_portfolio,
            new_portfolio=new_portfolio,
            time_stop_df=time_stop_df,
        )

    html_body = build_html_email(
        run_date=run_date,
        market_momentum=market_momentum,
        mode=mode,
        ranked=ranked,
        new_portfolio=new_portfolio,
        current_portfolio=current_portfolio,
        sells=sells,
        buys=buys,
        holds=holds,
        time_stop_tickers=time_stop_tickers,
        time_stop_df=time_stop_df,
    )

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_body)

    save_new_portfolio(new_portfolio)

    print("==============================")
    print("MONTHLY FTSE MOMENTUM REBALANCE")
    print("==============================")
    print(f"Run date: {run_date}")
    print(f"FTSE 350 blended momentum: {market_momentum:.2%}")
    print(f"Mode: {mode}")
    print(f"SELL: {', '.join(sells) if sells else 'None'}")
    print(f"BUY: {', '.join(buys) if buys else 'None'}")
    print(f"HOLD: {', '.join(holds) if holds else 'None'}")
    print(f"HTML email saved to: {HTML_FILE}")
    print(f"{PORTFOLIO_FILE} updated.")


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    run_monthly_rebalance()
