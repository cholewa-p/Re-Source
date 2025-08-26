CREATE TABLE clients (
    client_id SERIAL PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL
);
CREATE TABLE addresses (
    address_id SERIAL PRIMARY KEY,
    client_id INT NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
    street TEXT NOT NULL,
    city TEXT NOT NULL,
    postal_code TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE energy_sources (
    source_id SERIAL PRIMARY KEY,
    address_id INT NOT NULL REFERENCES addresses(address_id) ON DELETE CASCADE,
    source_type TEXT NOT NULL,  -- np. 'PV', 'Wind', 'Battery', 'Generator'
    capacity_kw NUMERIC(10,3)   -- opcjonalnie moc znamionowa źródła
);
-- Dane czasowe dla konkretnego źródła energii
CREATE TABLE power_generation (
    time TIMESTAMPTZ NOT NULL,
    source_id INT NOT NULL REFERENCES energy_sources(source_id) ON DELETE CASCADE,
    power_kw NUMERIC(10,3) NOT NULL,
    PRIMARY KEY (time, source_id)
);
---- Hypertable w Timescale
SELECT create_hypertable('power_generation', 'time');
CREATE TABLE user_accounts (
    account_id SERIAL PRIMARY KEY,
    client_id INT NOT NULL references clients(client_id) ON DELETE CASCADE,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);
CREATE TABLE models (
  model_id BIGSERIAL PRIMARY KEY,
  account_id INTEGER NOT NULL
    REFERENCES user_accounts(account_id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  model_path TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata JSONB
);

CREATE TABLE consumption_meters (
    meter_id SERIAL PRIMARY KEY,
    address_id INT NOT NULL,
    property_type VARCHAR(50),
    max_load_kw DECIMAL(8, 2)
);

CREATE TABLE consumption_readings (
    reading_id SERIAL PRIMARY KEY,
    meter_id INT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    consumption_kw DECIMAL(10, 3) NOT NULL,
    FOREIGN KEY (meter_id) REFERENCES consumption_meters (meter_id)
);
ALTER TABLE consumption_meters
ADD CONSTRAINT fk_consumption_meters_addresses
FOREIGN KEY (address_id) REFERENCES addresses (address_id);

