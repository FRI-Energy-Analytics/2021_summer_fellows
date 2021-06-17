/* Create tables and COPY data from csv files as strings.
*/
CREATE TABLE orphans(
    district VARCHAR,
    api VARCHAR, 
    ofcu_well_priority VARCHAR, 
    stat VARCHAR,
    operator_name VARCHAR,
    operator_no VARCHAR, 
    lease_name VARCHAR,
    lease_no VARCHAR, 
    well_no VARCHAR, 
	bonded_depth VARCHAR,
	field_name VARCHAR,
	county VARCHAR,
	sfp_code VARCHAR,
	ice_inspection_date VARCHAR,
	ice_inspection_id VARCHAR,
	sb639_enf VARCHAR,
	sb639_r15 VARCHAR,
    months_delinquent VARCHAR
)

CREATE TABLE completions(
    og_code VARCHAR,
    district_no VARCHAR, 
    lease_no VARCHAR, 
    well_no VARCHAR,
    api_county VARCHAR, 
    api_unique VARCHAR,
    onshore_assc_cnty VARCHAR, 
    district_name VARCHAR, 
	county_name VARCHAR,
	oil_well_unit_no VARCHAR,
	well_root_no VARCHAR,
	wellbore_shutin_dt VARCHAR,
	well_shutin_dt VARCHAR,
	well_14b2_status_code VARCHAR,
	well_subject_14b2_flag VARCHAR,
	wellbore_location_code VARCHAR
);


CREATE TABLE operators(
    operator_no VARCHAR,
    operator_name VARCHAR, 
    p5_status_code VARCHAR, 
    p5_last_filed_dt VARCHAR,
    oeprator_tax_cert_flag VARCHAR, 
    operator_sb639_flag VARCHAR,
    fa_option_code VARCHAR,
    record_status_code VARCHAR, 
    efile_status_code VARCHAR,
    efile_effective_dt VARCHAR,
    create_by VARCHAR,
    create_dt VARCHAR,
    modify_by VARCHAR,
    modify_dt VARCHAR
);

-- Up to 03/21 (Cushing, OK WTI Spot Price FOB (Dollars per Barrel))
CREATE TABLE oil_prices(
    date VARCHAR,
    price DOUBLE PRECISION
);

-- Up to 05/21 (Henry Hub Natural Gas Spot Price Dollars per Million BTU)
CREATE TABLE gas_prices(
    date VARCHAR,
    price DOUBLE PRECISION
);

-- Retrieved 05/08/21
CREATE TABLE iwar(
	operator_no VARCHAR,
	operator_name VARCHAR,
    api_county VARCHAR,
    api_unique VARCHAR, 
	county_name VARCHAR,
	og_code VARCHAR,
	district_code VARCHAR,
	lease_no VARCHAR,
	well_no VARCHAR,
	oil_unit_no VARCHAR,
	lease_name VARCHAR,
	field_no VARCHAR,
	field_name VARCHAR,
	water_land_code VARCHAR,
	api_depth VARCHAR,
	shutin_dt VARCHAR,
	p5_renewal_month VARCHAR,
	p5_renewal_year VARCHAR,
	p5_originating_status VARCHAR,
	current_5yr_inactive VARCHAR,
	current_10yr_inactive VARCHAR,
	aged_5yr_inactive VARCHAR, 
	aged_10yr_inactive VARCHAR,
	current_inactive_yrs VARCHAR,
	current_inactive_months VARCHAR,
	aged_inactive_yrs VARCHAR,
	aged_inactive_months VARCHAR,
	extension_status VARCHAR,
	ext_data_h VARCHAR,
	ext_data_e VARCHAR,
	ext_data_p VARCHAR,
	ext_data_f VARCHAR,
	ext_data_v VARCHAR,
	ext_data_x VARCHAR,
	ext_data_o VARCHAR,
	ext_data_t VARCHAR,
	ext_data_m VARCHAR,
	ext_data_k VARCHAR,
	ext_data_r VARCHAR,
	ext_data_q VARCHAR,
	ext_data_s VARCHAR,
	ext_data_w VARCHAR,
	cost_calc VARCHAR,
	well_plugged VARCHAR,
	compliance_due_date VARCHAR,
	orig_completion VARCHAR,
    phasein_5yr VARCHAR,
    dummy VARCHAR
);

-- Import data. Use \copy if access/permission is denied.
COPY orphans FROM 'orphansnew.txt' (DELIMITER('\t'));
COPY completions FROM 'well_completions.csv' DELIMITER ',' CSV HEADER;
COPY operators FROM 'operators.csv' DELIMITER ',' CSV HEADER;
COPY iwar FROM 'iwardata.txt' (DELIMITER('\t'));
COPY oil_prices FROM 'WTI_prices.csv' DELIMITER ',' CSV HEADER;
COPY gas_prices FROM 'Henry_Hub_Natural_Gas_Spot_Price.csv' DELIMITER ',' CSV HEADER;

-- Delete columns with only null values.
ALTER TABLE orphans DROP COLUMN stat;
ALTER TABLE operators DROP COLUMN record_status_code; 
ALTER TABLE operators DROP COLUMN efile_status_code; 
ALTER TABLE operators DROP COLUMN efile_effective_dt; 
ALTER TABLE operators DROP COLUMN modify_by; 
ALTER TABLE operators DROP COLUMN modify_dt; 
ALTER TABLE iwar DROP COLUMN phasein_5yr;
ALTER TABLE iwar DROP COLUMN dummy;
ALTER TABLE iwar ADD PRIMARY KEY(lease_name, lease_no, well_no);

-- Combine API county and unique codes in completions data.
ALTER TABLE completions ADD COLUMN api VARCHAR;
UPDATE completions SET api = api_county_code || api_unique_no;
ALTER TABLE completions DROP COLUMN api_county_code, DROP COLUMN api_unique_no;

-- Combine API county and unique codes in IWAR data.
ALTER TABLE iwar ADD COLUMN api varchar;
UPDATE iwar SET api_unique = LPAD(api_unique, 5, '0') WHERE LENGTH(api_unique) < 5;
UPDATE iwar SET api_county = LPAD(api_county, 3, '0') WHERE LENGTH(api_county) < 3;
UPDATE iwar SET api = api_county || api_unique;
ALTER TABLE iwar DROP COLUMN api_county, DROP COLUMN api_unique;

-- Standardize Date Formats (YYYY-MM-DD)
-- Convert oil prices dates
UPDATE oil_prices SET date = TO_DATE(date, 'DD-MM-YYYY');
ALTER TABLE oil_prices ALTER COLUMN date TYPE date USING date::date;

-- Convert gas prices dates
UPDATE gas_prices SET date = date || '-15'; --Setting monthly data to be on the 15th of each month
UPDATE gas_prices SET date = TO_DATE(date, 'Mon-YY-DD');
ALTER TABLE gas_prices ALTER COLUMN date TYPE date USING date::date;

-- Convert completions well shut-in dates
UPDATE completions SET wellbore_shutin_dt = wellbore_shutin_dt || '15';
UPDATE completions SET well_shutin_dt = well_shutin_dt || '15';
UPDATE completions SET wellbore_shutin_dt = TO_DATE(wellbore_shutin_dt, 'YYYYMMDD');
UPDATE completions SET well_shutin_dt = TO_DATE(well_shutin_dt, 'YYYYMMDD');
-- 0's in dataset were converted to BC dates, null out these values
UPDATE completions SET wellbore_shutin_dt = NULL WHERE wellbore_shutin_dt LIKE '%BC'; 
UPDATE completions SET well_shutin_dt = NULL WHERE well_shutin_dt LIKE '%BC';
ALTER TABLE completions ALTER COLUMN wellbore_shutin_dt TYPE date USING wellbore_shutin_dt::date;
ALTER TABLE completions ALTER COLUMN well_shutin_dt TYPE date USING well_shutin_dt::date;

-- Convert operators dates
UPDATE operators SET create_dt = TO_DATE(create_dt, 'DD-MON-YY')
UPDATE operators SET p5_last_filed_dt = TO_DATE(p5_last_filed_dt, 'YYYYMMDD');
UPDATE operators SET p5_last_filed_dt = NULL WHERE p5_last_filed_dt LIKE '%BC';
ALTER TABLE operators ALTER COLUMN create_dt TYPE date USING create_dt::date;
ALTER TABLE operators ALTER COLUMN p5_last_filed_dt TYPE date USING p5_last_filed_dt::date;

-- Convert IWAR dates
UPDATE iwar SET shutin_dt = shutin_dt || '15';
UPDATE iwar SET shutin_dt = TO_DATE(shutin_dt, 'YYYYMMDD');
ALTER TABLE iwar ALTER COLUMN shutin_dt TYPE date USING shutin_dt::date;
UPDATE iwar SET compliance_due_date = NULL WHERE compliance_due_date = '0';
UPDATE iwar SET compliance_due_date = TO_DATE(compliance_due_date, 'YYYYMMDD');
ALTER TABLE iwar ALTER COLUMN compliance_due_date TYPE date USING compliance_due_date::date;
-- Original completion dates
UPDATE iwar SET orig_completion = NULL WHERE orig_completion = '0';
ALTER TABLE iwar ADD COLUMN orig_completion_yr VARCHAR, ADD COLUMN orig_completion_month VARCHAR, ADD COLUMN orig_completion_day VARCHAR;
UPDATE iwar SET orig_completion_yr = SUBSTR(orig_completion, 1, 4);
UPDATE iwar SET orig_completion_month = SUBSTR(orig_completion, 5, 2);
UPDATE iwar SET orig_completion_day = SUBSTR(orig_completion, 7, 2);
UPDATE iwar SET orig_completion_month = '01' WHERE orig_completion_month = '00'
UPDATE iwar SET orig_completion_day = '01' WHERE orig_completion_day = '00' OR orig_completion_day::int > 31
UPDATE iwar SET orig_completion_day = '01' WHERE orig_completion_day = '00' OR SUBSTR(orig_completion, 5, 2) = '00';
UPDATE iwar SET orig_completion = orig_completion_yr || orig_completion_month || orig_completion_day;
UPDATE iwar SET orig_completion = TO_DATE(orig_completion, 'YYYYMMDD');
ALTER TABLE iwar ALTER COLUMN orig_completion TYPE date USING orig_completion::date;

-- Convert orphan list dates
UPDATE orphans SET ice_inspection_date = ice_inspection_date::date WHERE ice_inspection_date <> '';
UPDATE orphans SET ice_inspection_date = NULL WHERE ice_inspection_date = '';
ALTER TABLE orphans ALTER COLUMN ice_inspection_date TYPE date USING ice_inspection_date::date;

-- Cleaning lease/operator names in IWAR data
UPDATE iwar SET lease_name = SUBSTR(lease_name, 2, LENGTH(lease_name) - 2) WHERE LEFT(lease_name, 1) = '"' AND RIGHT(lease_name, 1) = '"';
UPDATE iwar SET lease_name = REGEXP_REPLACE(lease_name, '\s+$', '');
UPDATE iwar SET operator_name = SUBSTR(operator_name, 2, LENGTH(operator_name) - 2) WHERE LEFT(operator_name, 1) = '"' AND RIGHT(operator_name, 1) = '"';
UPDATE iwar SET operator_name = REGEXP_REPLACE(operator_name, '\s+$', '');
UPDATE iwar SET field_name = SUBSTR(field_name, 2, LENGTH(field_name) - 2) WHERE LEFT(field_name, 1) = '"' AND RIGHT(field_name, 1) = '"';
UPDATE iwar SET field_name = REGEXP_REPLACE(field_name, '\s+$', '');
UPDATE iwar SET lease_no = LPAD(lease_no, 6, '0') WHERE LENGTH(lease_no) < 6 AND og_code = 'G';
UPDATE iwar SET lease_no = LPAD(lease_no, 5, '0') WHERE LENGTH(lease_no) < 5 AND og_code = 'O';

-- Fix lease numbers in IWAR data
UPDATE iwar SET lease_no = LPAD(lease_no, 6, '0') WHERE LENGTH(lease_no) < 6 AND og_code = 'G';
UPDATE iwar SET lease_no = LPAD(lease_no, 5, '0') WHERE LENGTH(lease_no) < 5 AND og_code = 'O';

-- Cleaning operator/field names in orphans data
UPDATE orphans SET operator_name = SUBSTR(operator_name, 2, LENGTH(operator_name) - 2) WHERE LEFT(operator_name, 1) = '"' AND RIGHT(operator_name, 1) = '"';
UPDATE orphans SET operator_name = REGEXP_REPLACE(operator_name, '\s+$', '');
UPDATE orphans SET field_name = SUBSTR(field_name, 2, LENGTH(field_name) - 2) WHERE LEFT(field_name, 1) = '"' AND RIGHT(field_name, 1) = '"';
UPDATE orphans SET field_name = REGEXP_REPLACE(field_name, '\s+$', '');

-- Remove white spaces from well numbers
UPDATE orphans SET well_no = REPLACE(well_no, ' ', '');
UPDATE iwar SET well_no = REPLACE(well_no, ' ', '');

-- Adjusting data types
ALTER TABLE orphans ALTER COLUMN bonded_depth TYPE int USING bonded_depth::integer;
ALTER TABLE orphans ALTER COLUMN months_delinquent TYPE int USING months_delinquent::integer;
ALTER TABLE iwar ALTER COLUMN api_depth TYPE int USING api_depth::integer;
ALTER TABLE iwar ALTER COLUMN p5_renewal_month TYPE int USING p5_renewal_month::integer;
ALTER TABLE iwar ALTER COLUMN p5_renewal_year TYPE int USING p5_renewal_year::integer;
ALTER TABLE iwar ALTER COLUMN current_inactive_yrs TYPE int USING current_inactive_yrs::integer;
ALTER TABLE iwar ALTER COLUMN current_inactive_months TYPE int USING current_inactive_months::integer;
ALTER TABLE iwar ALTER COLUMN aged_inactive_yrs TYPE int USING aged_inactive_yrs::integer;
ALTER TABLE iwar ALTER COLUMN aged_inactive_months TYPE int USING aged_inactive_months::integer;
ALTER TABLE iwar ALTER COLUMN cost_calc TYPE int USING cost_calc::integer;

-- Setting blank values to null
UPDATE iwar SET oil_unit_no = NULL WHERE oil_unit_no = ' ';
UPDATE iwar SET current_5yr_inactive = NULL WHERE current_5yr_inactive = ' ';
UPDATE iwar SET current_10yr_inactive = NULL WHERE current_10yr_inactive = ' ';
UPDATE iwar SET aged_5yr_inactive = NULL WHERE aged_5yr_inactive = ' ';
UPDATE iwar SET aged_10yr_inactive = NULL WHERE aged_10yr_inactive = ' ';
UPDATE iwar SET api_depth = NULL WHERE api_depth = 0;
UPDATE iwar SET p5_renewal_month = NULL WHERE p5_renewal_month = 0;
UPDATE iwar SET cost_calc = NULL WHERE cost_calc = 0;
UPDATE orphans SET ofcu_well_priority = NULL WHERE ofcu_well_priority = '';
UPDATE orphans SET sfp_code = NULL WHERE sfp_code = '';
UPDATE orphans SET ice_inspection_id = NULL WHERE ice_inspection_id = '';
UPDATE orphans SET sb639_enf = NULL WHERE sb639_enf = '';
UPDATE orphans SET sb639_r15 = NULL WHERE sb639_r15 = '';
UPDATE orphans SET bonded_depth = NULL WHERE bonded_depth = 0;

-- Set columns that should not be null
ALTER TABLE iwar ALTER COLUMN operator_no SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN operator_name SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN county_name SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN og_code SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN district_code SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN lease_no SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN well_no SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN field_no SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN field_name SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN water_land_code SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN shutin_dt SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN p5_renewal_year SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN p5_originating_status SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN well_plugged SET NOT NULL;
ALTER TABLE iwar ALTER COLUMN api SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN district SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN api SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN operator_name SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN operator_no SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN lease_name SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN lease_no SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN well_no SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN field_name SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN county SET NOT NULL;
ALTER TABLE orphans ALTER COLUMN months_delinquent SET NOT NULL;

-- Convert certain columns to boolean
UPDATE iwar SET ext_data_h = 't' WHERE ext_data_h IS NOT NULL;
UPDATE iwar SET ext_data_h = 'f' WHERE ext_data_h IS NULL;
UPDATE iwar SET ext_data_e = 't' WHERE ext_data_e IS NOT NULL;
UPDATE iwar SET ext_data_e = 'f' WHERE ext_data_e IS NULL;
UPDATE iwar SET ext_data_p = 't' WHERE ext_data_p IS NOT NULL;
UPDATE iwar SET ext_data_p = 'f' WHERE ext_data_p IS NULL;
UPDATE iwar SET ext_data_f = 't' WHERE ext_data_f IS NOT NULL;
UPDATE iwar SET ext_data_f = 'f' WHERE ext_data_f IS NULL;
UPDATE iwar SET ext_data_v = 't' WHERE ext_data_v IS NOT NULL;
UPDATE iwar SET ext_data_v = 'f' WHERE ext_data_v IS NULL;
UPDATE iwar SET ext_data_x = 't' WHERE ext_data_x IS NOT NULL;
UPDATE iwar SET ext_data_x = 'f' WHERE ext_data_x IS NULL;
UPDATE iwar SET ext_data_o = 't' WHERE ext_data_o IS NOT NULL;
UPDATE iwar SET ext_data_o = 'f' WHERE ext_data_o IS NULL;
UPDATE iwar SET ext_data_t = 't' WHERE ext_data_t IS NOT NULL;
UPDATE iwar SET ext_data_t = 'f' WHERE ext_data_t IS NULL;
UPDATE iwar SET ext_data_m = 't' WHERE ext_data_m IS NOT NULL;
UPDATE iwar SET ext_data_m = 'f' WHERE ext_data_m IS NULL;
UPDATE iwar SET ext_data_k = 't' WHERE ext_data_k IS NOT NULL;
UPDATE iwar SET ext_data_k = 'f' WHERE ext_data_k IS NULL;
UPDATE iwar SET ext_data_r = 't' WHERE ext_data_r IS NOT NULL;
UPDATE iwar SET ext_data_r = 'f' WHERE ext_data_r IS NULL;
UPDATE iwar SET ext_data_q = 't' WHERE ext_data_q IS NOT NULL;
UPDATE iwar SET ext_data_q = 'f' WHERE ext_data_q IS NULL;
UPDATE iwar SET ext_data_s = 't' WHERE ext_data_s IS NOT NULL;
UPDATE iwar SET ext_data_s = 'f' WHERE ext_data_s IS NULL;
UPDATE iwar SET ext_data_w = 't' WHERE ext_data_w IS NOT NULL;
UPDATE iwar SET ext_data_w = 'f' WHERE ext_data_w IS NULL;
UPDATE iwar SET current_5yr_inactive = 't' WHERE current_5yr_inactive IS NOT NULL;
UPDATE iwar SET current_5yr_inactive = 'f' WHERE current_5yr_inactive IS NULL;
UPDATE iwar SET current_10yr_inactive = 't' WHERE current_10yr_inactive IS NOT NULL;
UPDATE iwar SET current_10yr_inactive = 'f' WHERE current_10yr_inactive IS NULL;
UPDATE iwar SET aged_5yr_inactive = 't' WHERE aged_5yr_inactive IS NOT NULL;
UPDATE iwar SET aged_5yr_inactive = 'f' WHERE aged_5yr_inactive IS NULL;
UPDATE iwar SET aged_10yr_inactive = 't' WHERE aged_10yr_inactive IS NOT NULL;
UPDATE iwar SET aged_10yr_inactive = 'f' WHERE aged_10yr_inactive IS NULL;
UPDATE iwar SET well_plugged = 't' WHERE well_plugged = 'Y';
UPDATE iwar SET well_plugged = 'f' WHERE well_plugged = 'N';
ALTER TABLE iwar ALTER COLUMN ext_data_h TYPE boolean USING ext_data_h::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_e TYPE boolean USING ext_data_e::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_p TYPE boolean USING ext_data_p::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_f TYPE boolean USING ext_data_f::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_v TYPE boolean USING ext_data_v::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_x TYPE boolean USING ext_data_x::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_o TYPE boolean USING ext_data_o::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_t TYPE boolean USING ext_data_t::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_m TYPE boolean USING ext_data_m::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_k TYPE boolean USING ext_data_k::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_r TYPE boolean USING ext_data_r::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_q TYPE boolean USING ext_data_q::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_s TYPE boolean USING ext_data_s::boolean;
ALTER TABLE iwar ALTER COLUMN ext_data_w TYPE boolean USING ext_data_w::boolean;
ALTER TABLE iwar ALTER COLUMN current_5yr_inactive TYPE boolean USING current_5yr_inactive::boolean;
ALTER TABLE iwar ALTER COLUMN current_10yr_inactive TYPE boolean USING current_10yr_inactive::boolean; 
ALTER TABLE iwar ALTER COLUMN aged_5yr_inactive TYPE boolean USING aged_5yr_inactive::boolean; 
ALTER TABLE iwar ALTER COLUMN aged_10yr_inactive TYPE boolean USING aged_10yr_inactive::boolean; 
ALTER TABLE iwar ALTER COLUMN well_plugged TYPE boolean USING well_plugged::boolean;

-- Query to join IWAR and orphan wells list based on lease name, lease id, and well number
SELECT iwar.og_code, orphans.operator_name, orphans.operator_no, orphans.lease_name, orphans.lease_no, 
orphans.well_no, orphans.api, orphans.district, orphans.county, orphans.field_name, iwar.field_no, 
iwar.oil_unit_no, iwar.water_land_code, iwar.api_depth, iwar.shutin_dt, iwar.well_plugged, iwar.cost_calc, 
iwar.compliance_due_date,  iwar.orig_completion, orphans.ice_inspection_date, orphans.ice_inspection_id, 
orphans.sfp_code, orphans.ofcu_well_priority, orphans.sb639_enf, orphans.sb639_r15, orphans.months_delinquent, 
iwar.p5_renewal_month, iwar.p5_renewal_year, iwar.p5_originating_status, iwar.current_5yr_inactive, 
iwar.current_10yr_inactive, iwar.aged_5yr_inactive, iwar.aged_10yr_inactive, iwar.current_inactive_yrs, 
iwar.current_inactive_months, iwar.aged_inactive_yrs, iwar.aged_inactive_months, iwar.extension_status, 
iwar.ext_data_h, iwar.ext_data_e, iwar.ext_data_p, iwar.ext_data_f, iwar.ext_data_v, iwar.ext_data_x, 
iwar.ext_data_o, iwar.ext_data_t, iwar.ext_data_m, iwar.ext_data_k, iwar.ext_data_r, iwar.ext_data_q, 
iwar.ext_data_s, iwar.ext_data_w FROM orphans JOIN iwar ON (orphans.lease_name = iwar.lease_name AND 
orphans.lease_no = iwar.lease_no AND orphans.well_no = iwar.well_no) WHERE iwar.orig_completion IS NOT NULL;

-- Query to analyze repeat well API numbers in completions data
/*SELECT * FROM (
	SELECT api, count(*)
	FROM completions
	GROUP BY api
	HAVING count(api) > 1) tbl JOIN completions ON completions.api = tbl.api
	ORDER BY completions.api; */
