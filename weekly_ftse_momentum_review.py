# weekly_ftse_momentum_review.py

import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from datetime import datetime

# ======================================================
# SETTINGS
# ======================================================

TOP_N = 10

PRICE_SCALE = 0.01  # Yahoo UK .L prices are usually quoted in pence

MIN_PRICE_GBP = 0.50
MIN_AVG_VALUE_TRADED_GBP = 1_000_000

EXCLUDED_SECTORS = ["investment trusts"]

LOOKBACK_LONG_DAYS = 84      # approx 4 months
LOOKBACK_SHORT_DAYS = 42     # approx 2 months
SKIP_DAYS = 21               # exclude most recent month from long momentum

WEIGHT_SHORT = 0.50
WEIGHT_LONG = 0.50

MAX_ALLOWED_MOMENTUM = 2.5
MAX_SINGLE_DAY_MOVE = 0.50
MAX_ANNUALISED_VOL = 1.80

MIN_VALID_DAYS = max(LOOKBACK_LONG_DAYS, LOOKBACK_SHORT_DAYS, SKIP_DAYS)

MARKET_PROXY = "^FTLC"
HTML_FILE = "ftse_weekly_email.html"


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
# TOP PORTFOLIO
# ======================================================

def calculate_top_portfolio(data, ftse350_table):
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
            "Momentum2M": components["Momentum2M"],
            "Momentum4MEx1M": components["Momentum4MEx1M"],
            "BlendedMomentum": components["BlendedMomentum"],
        })

    ranked = pd.DataFrame(results)

    if ranked.empty:
        return pd.DataFrame(), len(results), rejected_count

    ranked = ranked.sort_values("BlendedMomentum", ascending=False).reset_index(drop=True)
    ranked["RawRank"] = ranked.index + 1

    top_portfolio = ranked.head(TOP_N).copy()
    top_portfolio = top_portfolio.reset_index(drop=True)
    top_portfolio["PortfolioRank"] = top_portfolio.index + 1

    return top_portfolio, len(results), rejected_count


# ======================================================
# HTML EMAIL
# ======================================================

def build_html_email(run_date, market_momentum, mode, top_portfolio):
    if mode == "RISK-OFF":
        header_colour = "#b00020"
        action = "ACTION REQUIRED: CLOSE ALL FTSE POSITIONS"
        summary = (
            "FTSE 350 2/4 blended momentum is negative or zero. "
            "The FTSE system is now risk-off."
        )
    else:
        header_colour = "#0b6b3a"
        action = "RISK-ON: HOLD CURRENT FTSE POSITIONS"
        summary = (
            "FTSE 350 2/4 blended momentum remains positive. "
            "No weekly trading action is required."
        )

    rows = ""

    if top_portfolio is not None and not top_portfolio.empty:
        for _, row in top_portfolio.iterrows():
            rows += f"""
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

    table_html = ""

    if rows:
        table_html = f"""
        <h2>Current FTSE 350 Top 10 Momentum Leaders</h2>
        <table>
            <tr>
                <th>Portfolio Rank</th>
                <th>Raw Rank</th>
                <th>Ticker</th>
                <th>Name</th>
                <th>Sector</th>
                <th>2M Momentum</th>
                <th>4M Ex-1M</th>
                <th>Blended Momentum</th>
            </tr>
            {rows}
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
            .status {{
                font-size: 18px;
                font-weight: bold;
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
                <h1>Weekly FTSE Momentum Review</h1>
                <h2>{action}</h2>
            </div>

            <div class="box">
                <p><strong>Run date:</strong> {run_date}</p>
                <p><strong>Market proxy:</strong> {MARKET_PROXY}</p>
                <p><strong>FTSE 350 blended momentum:</strong> {market_momentum:.2%}</p>
                <p><strong>Mode:</strong> <span class="status">{mode}</span></p>
                <p>{summary}</p>
            </div>

            {table_html}

            <div class="footer">
                <p>
                    This is a weekly review only. Monthly rebalance decisions remain separate.
                    The only weekly action is to close all FTSE positions if FTSE 350 blended momentum is negative or zero.
                </p>
            </div>

        </div>
    </body>
    </html>
    """


# ======================================================
# MAIN REVIEW
# ======================================================

def run_review():
    run_date = datetime.now().strftime("%Y-%m-%d")

    ftse350_table = get_ftse350_table()
    tickers = ftse350_table["Ticker"].tolist()

    all_tickers = sorted(list(set(tickers + [MARKET_PROXY])))

    data = download_data(all_tickers)

    market_df = get_ticker_data(data, MARKET_PROXY)

    if market_df is None or market_df.empty:
        raise ValueError("Could not retrieve FTSE 350 market proxy data.")

    market_components = calculate_momentum_components(market_df["Close"])

    if market_components is None:
        raise ValueError("Could not calculate FTSE 350 blended momentum.")

    market_momentum = market_components["BlendedMomentum"]
    mode = "RISK-ON" if market_momentum > 0 else "RISK-OFF"

    top_portfolio, qualified_count, rejected_count = calculate_top_portfolio(
        data=data,
        ftse350_table=ftse350_table,
    )

    html_body = build_html_email(
        run_date=run_date,
        market_momentum=market_momentum,
        mode=mode,
        top_portfolio=top_portfolio,
    )

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_body)

    print("==============================")
    print("WEEKLY FTSE MOMENTUM REVIEW")
    print("==============================")
    print(f"Run date: {run_date}")
    print(f"FTSE 350 blended momentum: {market_momentum:.2%}")
    print(f"Mode: {mode}")
    print(f"Qualified stocks: {qualified_count}")
    print(f"Rejected stocks: {rejected_count}")
    print(f"HTML email saved to: {HTML_FILE}")

    if mode == "RISK-OFF":
        print("ACTION REQUIRED: CLOSE ALL FTSE POSITIONS")
    else:
        print("RISK-ON: HOLD CURRENT FTSE POSITIONS")


# ======================================================
# RUN
# ======================================================

if __name__ == "__main__":
    run_review()
