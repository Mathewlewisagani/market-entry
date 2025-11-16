CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    UNIQUE (symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_timestamp
    ON stock_prices (symbol, timestamp);

CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    rsi NUMERIC,
    macd NUMERIC,
    signal_line NUMERIC,
    sma_20 NUMERIC,
    sma_50 NUMERIC,
    sma_200 NUMERIC,
    bollinger_upper NUMERIC,
    bollinger_lower NUMERIC,
    UNIQUE (symbol, timestamp),
    FOREIGN KEY (symbol, timestamp)
        REFERENCES stock_prices (symbol, timestamp)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_timestamp
    ON technical_indicators (symbol, timestamp);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    prediction_date TIMESTAMP,
    signal VARCHAR,
    confidence NUMERIC,
    model_version VARCHAR,
    features JSONB,
    UNIQUE (symbol, prediction_date)
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_prediction_date
    ON ml_predictions (symbol, prediction_date);

CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR,
    version VARCHAR,
    trained_at TIMESTAMP,
    accuracy NUMERIC,
    parameters JSONB
);
