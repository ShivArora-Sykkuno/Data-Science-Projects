CREATE TABLE daily_stats (
    id SERIAL PRIMARY KEY,
    client_id VARCHAR(255),
    platform VARCHAR(50),
    date DATE,
    followers INT,
    impressions INT,
    engagement_rate FLOAT
);