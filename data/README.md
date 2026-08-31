# data/

The scripts expect SQLite databases in this folder. The full databases are not
committed (`.gitignore`); `sample_sol_5min.db` is a real 1,081-candle extract
(SOL/USDC, 5-minute, January 2026) that carries both schemas so any script can
be pointed at it for a quick run.

## Schemas

**2026 schema** — built by `market-data-pipeline/data_pipeline/binance_chart_scraper.py`. One table per
market and timeframe in `crypto.db` (`solana`, `solana1H`, `Bitcoin1H`, `USDCUSDT1H`, …):

| column   | type    | notes                                                        |
|----------|---------|--------------------------------------------------------------|
| position | INTEGER | candles since the oldest row — gaps show up as jumps         |
| date     | STRING  | `YYYY/MM/DD HH:MM`                                           |
| open, high, low, close | FLOAT |                                                |
| volume   | FLOAT   | base-asset volume as displayed on the chart                  |

**2025 schema** — built by the 2025 scraper in the pipeline repo in
`SQL001.db` (`Solana`, `Solana3`, `BTC`, `ETH`): `date TEXT, high, low, open, close`
(no position, no volume; note the column order). Rows are addressed by ROWID.

## Using the sample

Set the database/table constants at the top of a script to the sample
(`DATABASE = "sample_sol_5min.db"`, `TABLENAME = "solana"` for the 2026 schema;
`DB_FILE = "data/sample_sol_5min.db"`, `TABLE_NAME = "Solana3"` for the 2025 one)
and shrink any row-window constants (`OFFSET`, `LENGTH`, `START_ROW`/`END_ROW`)
to fit 1,081 rows. The studies were run on 90,000–125,000-row tables.
