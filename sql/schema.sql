CREATE TABLE IF NOT EXISTS visits (
  trail_id integer,
  user_id integer,
  venue_id text,
  venue_category text,
  venue_schema text,
  venue_city text,
  venue_country text,
  "timestamp" timestamptz,
  trail_id_orig text,
  user_id_orig text
);
CREATE INDEX IF NOT EXISTS idx_visits_venue ON visits(venue_id);
CREATE INDEX IF NOT EXISTS idx_visits_city ON visits(venue_city);

CREATE TABLE IF NOT EXISTS pois (
  fsq_id text PRIMARY KEY,
  name text,
  lat double precision,
  lon double precision,
  city text,
  country text,
  rating double precision,
  price_tier integer,
  total_ratings integer,
  primary_category text,
  is_free boolean DEFAULT false
);
CREATE INDEX IF NOT EXISTS idx_pois_city ON pois(city);
CREATE INDEX IF NOT EXISTS idx_pois_primary_cat ON pois(primary_category);

CREATE TABLE IF NOT EXISTS poi_categories (
  fsq_id text REFERENCES pois(fsq_id) ON DELETE CASCADE,
  category_id text,
  category_name text,
  PRIMARY KEY (fsq_id, category_id)
);
