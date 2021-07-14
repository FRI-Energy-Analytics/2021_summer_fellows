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
-- Source: U.S. Energy Information Administration
CREATE TABLE oil_prices(
    date VARCHAR,
    price DOUBLE PRECISION
);

-- Up to 05/21 (Henry Hub Natural Gas Spot Price Dollars per Million BTU)
-- Source: U.S. Energy Information Administration
CREATE TABLE gas_prices(
    date VARCHAR,
    price DOUBLE PRECISION
);

-- Retrieved 05/08/21
CREATE TABLE inactive_wells(
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

-- From Texas RRC. Lease info connected to operators, fields, counties, and districts.
CREATE TABLE leases(
	og_code VARCHAR,
	district_no VARCHAR,
	lease_no VARCHAR,
	district_name VARCHAR,
	lease_name VARCHAR,
	operator_no VARCHAR,
	operator_name VARCHAR,
	field_no VARCHAR,
	field_name VARCHAR,
	well_no VARCHAR,
	lease_off_sched_flag VARCHAR,
	lease_severance_flag VARCHAR
)

-- Production data
-- By county:
CREATE TABLE county_prod (
	county_no VARCHAR,
	district_no VARCHAR,
	cycle_year VARCHAR,
	cycle_month VARCHAR,
	cycle_year_month VARCHAR,
	cnty_oil_prod_vol VARCHAR,
	cnty_gas_prod_vol VARCHAR,
	cnty_cond_prod_vol VARCHAR,
	cnty_csgd_prod_vol VARCHAR,
	county_name VARCHAR,
	district_name VARCHAR,
	og_code VARCHAR
)

-- By operator:
CREATE TABLE operator_prod (
	operator_no VARCHAR,
	cycle_year VARCHAR,
	cycle_month VARCHAR,
	cycle_year_month VARCHAR,
	operator_name VARCHAR,
	operator_oil_prod_vol VARCHAR,
	operator_gas_prod_vol VARCHAR,
	operator_cond_prod_vol VARCHAR,
	opreator_csgd_prod_vol VARCHAR
);

-- By district:
CREATE TABLE district_prod (
	district_no VARCHAR,
	cycle_year VARCHAR,
	cycle_month VARCHAR,
	cycle_year_month VARCHAR,
	district_name VARCHAR,
	dist_oil_prod_vol VARCHAR,
	dist_gas_prod_vol VARCHAR,
	dist_cond_prod_vol VARCHAR,
	dist_csgd_prod_vol VARCHAR
);

-- All wellbores
CREATE TABLE wellbores (
	district VARCHAR,
	county_no VARCHAR,
	api VARCHAR,
	county_name VARCHAR,
	og_code  VARCHAR,
	lease_name VARCHAR,
	field_no VARCHAR,
	field_name VARCHAR,
	lease_no VARCHAR,
	well_no_display VARCHAR,
	oil_unit_no VARCHAR,
	operator_name VARCHAR,
	operator_no VARCHAR,
	water_land_code VARCHAR,
	multi_comp_flag VARCHAR,
	api_depth VARCHAR,
	wb_shutin_dt VARCHAR,
	wb_14b2_flag VARCHAR,
	well_type_name VARCHAR,
	well_shutin_dt VARCHAR,
	plug_date VARCHAR,
	plug_lease_name VARCHAR,
	plug_operator_name VARCHAR,
	recent_permit VARCHAR,
	recent_permit_lease_name VARCHAR,
	recent_permit_operator_no VARCHAR,
	on_schedule VARCHAR,
	og_wellbore_ewa_id VARCHAR,
	w2g1_filed_date VARCHAR,
	w2g1_date VARCHAR,
	completion_date VARCHAR,
	w3_file_date VARCHAR,
	created_by VARCHAR,
	created_dt VARCHAR,
	modified_by VARCHAR,
	modified_dt VARCHAR,
	well_no VARCHAR,
	p5_renewal_month VARCHAR,
	p5_renewal_year VARCHAR,
	p5_org_status VARCHAR,
	current_inactive_yrs VARCHAR,
	current_inactive_months VARCHAR,
	wl_14b2_ext_status VARCHAR,
	wl_14b2_mech_integ VARCHAR,
	wl_14b2_plg_ord_sf VARCHAR,
	wl_14b2_pollution VARCHAR,
	wl_14b2_fldops_hold VARCHAR,
	wl_14b2_h15_prob VARCHAR,
	wl_14b2_h15_delq VARCHAR,
	wl_14b2_oper_delq VARCHAR,
	wl_14b2_dist_sfp VARCHAR,
	wl_14b2_dist_sf_clnup VARCHAR,
	wl_14b2_dist_st_plg VARCHAR,
	wl_14b2_good_faith VARCHAR,
	wl_14b2_well_other VARCHAR,
	surf_eqp_viol VARCHAR,
	w3x_viol VARCHAR,
	h15_status_code VARCHAR,
	orig_completion_dt VARCHAR
)

-- More wellbore info
CREATE TABLE wellbores2 (
	api_county VARCHAR,
	api_unique VARCHAR,
	district VARCHAR,
	county_no VARCHAR,
	orig_completion_cc VARCHAR,
	orig_completion VARCHAR,
	api_depth VARCHAR,
	valid_fluid_level VARCHAR,
	cert_revoked_date VARCHAR,
	cert_denial_date VARCHAR,
	denial_reason VARCHAR,
	error_api_assign_code VARCHAR,
	refer_correct_api_num VARCHAR,
	dummy_api_num VARCHAR,
	date_dummy_replaced VARCHAR,
	newest_drl_pmt_num VARCHAR,
	cancel_expire_code VARCHAR,
	except_13A VARCHAR,
	fresh_water_flag VARCHAR,
	plug_flag VARCHAR,
	previous_api_num VARCHAR,
	completion_data_ind VARCHAR,
	hist_date_source_flag VARCHAR,
	ex14b2_count VARCHAR,
	designation_hb_1975_flag VARCHAR,
	designation_effective_dt VARCHAR,
	designation_revised_dt VARCHAR,
	designation_letter_dt VARCHAR,
	cert_effect_dt VARCHAR,
	water_land_code VARCHAR,
	bonded_depth VARCHAR,
	override_est_plug_cost VARCHAR,
	shutin_dt VARCHAR,
	override_bonded_depth VARCHAR,
	well_subject_14b2_flag VARCHAR,
	well_pend_removal_14b2_flag VARCHAR,
	orphan_well_hold_flag VARCHAR,
	w3x_well_flag VARCHAR
)

-- More operator information
CREATE TABLE p5_orgs (
	operator_no VARCHAR,
	operator_name VARCHAR,
	refiling_req_flag VARCHAR,
	p5_status_code VARCHAR,
	hold_mail_code VARCHAR,
	renewal_letter_code VARCHAR,
	organization_code VARCHAR,
	organ_other_comment VARCHAR,
	gatherer_code VARCHAR,
	org_address_line1 VARCHAR,
	org_address_line2 VARCHAR,
	org_city VARCHAR,
	org_state VARCHAR,
	org_zip VARCHAR,
	org_zip_suffix VARCHAR,
	location_address_line1 VARCHAR,
	location_address_line2 VARCHAR,
	location_city VARCHAR,
	location_state VARCHAR,
	location_zip VARCHAR,
	location_zip_suffix VARCHAR,
	date_built VARCHAR,
	date_inactive VARCHAR,
	org_phone_num VARCHAR,
	refile_notice_month VARCHAR,
	refile_letter_date VARCHAR,
	refile_notice_date VARCHAR,
	refile_received_date VARCHAR,
	last_p5_received_date VARCHAR,
	other_org_no VARCHAR,
	filing_problem_date VARCHAR,
	filing_problem_ltr_code VARCHAR,
	telephone_verify_flag VARCHAR,
	op_num_multi_used_flag VARCHAR,
	oil_gatherer_status VARCHAR,
	gas_gatherer_status VARCHAR,
	tax_cert VARCHAR,
	emergency_phone_num VARCHAR
)

-- Field data
CREATE TABLE field_data (
	field_no VARCHAR,
	field_name VARCHAR,
	district_no VARCHAR,
	district_name VARCHAR, 
	field_class VARCHAR,
	field_h2s_flag VARCHAR,
	field_manual_rev_flag VARCHAR,
	wildcat_flag VARCHAR,
	o_derived_rule_type_code VARCHAR,
	g_derived_rule_type_code VARCHAR,
	o_rescind_dt VARCHAR,
	g_rescind_dt VARCHAR,
	o_salt_dome_flag VARCHAR,
	g_salt_dome_flag VARCHAR,
	o_offshore_code VARCHAR,
	g_offshore_code VARCHAR,
	o_dont_permit VARCHAR,
	g_dont_permit VARCHAR,
	o_noa_man_rev_rule VARCHAR,
	g_noa_man_rev_rule VARCHAR,
	o_county_no VARCHAR,
	g_county_no VARCHAR,
	o_discovery_dt VARCHAR,
	g_discovery_dt VARCHAR,
	o_sched_remarks VARCHAR,
	g_sched_remarks VARCHAR,
	o_comments VARCHAR,
	g_comments VARCHAR
)

-- Texas Geologic Units (2005)
CREATE TABLE tx_geo_units (
	state VARCHAR,
	orig_lab VARCHAR,
	map_sym1 VARCHAR,
	map_sym2 VARCHAR,
	unit_link VARCHAR,
	prov_no VARCHAR,
	province VARCHAR,
	unit_name VARCHAR,
	unit_age VARCHAR,
	unit_desc VARCHAR,
	strat_unit VARCHAR,
	unit_com VARCHAR,
	map_ref VARCHAR,
	rocktype1 VARCHAR,
	rocktype2 VARCHAR,
	rocktype3 VARCHAR,
	unit_ref VARCHAR
)

-- Import data. Use \copy if access/permission is denied.
COPY orphans FROM 'orphansnew.txt' (DELIMITER('\t'));
COPY completions FROM 'well_completions.csv' DELIMITER ',' CSV HEADER;
COPY operators FROM 'operators.csv' DELIMITER ',' CSV HEADER;
COPY inactive_wells FROM 'inactive_wells_data.txt' (DELIMITER('\t'));
COPY oil_prices FROM 'WTI_prices.csv' DELIMITER ',' CSV HEADER;
COPY gas_prices FROM 'Henry_Hub_Natural_Gas_Spot_Price.csv' DELIMITER ',' CSV HEADER;
COPY leases FROM 'lease_info.csv' DELIMITER ',' CSV HEADER;
COPY county_prod FROM 'county_prod.csv' DELIMITER ',' CSV HEADER;
COPY operator_prod FROM 'operator_prod.csv' DELIMITER ',' CSV HEADER;
COPY district_prod FROM 'district_prod.csv' DELIMITER ',' CSV HEADER;
COPY wellbores FROM 'wellbores.csv' DELIMITER ',' CSV HEADER;


-- Delete columns with only null values.
ALTER TABLE orphans DROP COLUMN stat;
ALTER TABLE operators DROP COLUMN record_status_code; 
ALTER TABLE operators DROP COLUMN efile_status_code; 
ALTER TABLE operators DROP COLUMN efile_effective_dt; 
ALTER TABLE operators DROP COLUMN modify_by; 
ALTER TABLE operators DROP COLUMN modify_dt; 
ALTER TABLE inactive_wells DROP COLUMN phasein_5yr;
ALTER TABLE inactive_wells DROP COLUMN dummy;
ALTER TABLE inactive_wells ADD PRIMARY KEY(lease_name, lease_no, well_no);

-- Combine API county and unique codes in completions data.
ALTER TABLE completions ADD COLUMN api VARCHAR;
UPDATE completions SET api = api_county_code || api_unique_no;
ALTER TABLE completions DROP COLUMN api_county_code, DROP COLUMN api_unique_no;

-- Combine API county and unique codes in inactive_wells data.
ALTER TABLE inactive_wells ADD COLUMN api varchar;
UPDATE inactive_wells SET api_unique = LPAD(api_unique, 5, '0') WHERE LENGTH(api_unique) < 5;
UPDATE inactive_wells SET api_county = LPAD(api_county, 3, '0') WHERE LENGTH(api_county) < 3;
UPDATE inactive_wells SET api = api_county || api_unique;
ALTER TABLE inactive_wells DROP COLUMN api_county, DROP COLUMN api_unique;

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

-- Convert inactive_wells dates
UPDATE inactive_wells SET shutin_dt = shutin_dt || '15';
UPDATE inactive_wells SET shutin_dt = TO_DATE(shutin_dt, 'YYYYMMDD');
ALTER TABLE inactive_wells ALTER COLUMN shutin_dt TYPE date USING shutin_dt::date;
UPDATE inactive_wells SET compliance_due_date = NULL WHERE compliance_due_date = '0';
UPDATE inactive_wells SET compliance_due_date = TO_DATE(compliance_due_date, 'YYYYMMDD');
ALTER TABLE inactive_wells ALTER COLUMN compliance_due_date TYPE date USING compliance_due_date::date;
-- Original completion dates
UPDATE inactive_wells SET orig_completion = NULL WHERE orig_completion = '0';
ALTER TABLE inactive_wells ADD COLUMN orig_completion_yr VARCHAR, ADD COLUMN orig_completion_month VARCHAR, ADD COLUMN orig_completion_day VARCHAR;
UPDATE inactive_wells SET orig_completion_yr = SUBSTR(orig_completion, 1, 4);
UPDATE inactive_wells SET orig_completion_month = SUBSTR(orig_completion, 5, 2);
UPDATE inactive_wells SET orig_completion_day = SUBSTR(orig_completion, 7, 2);
UPDATE inactive_wells SET orig_completion_month = '01' WHERE orig_completion_month = '00'
UPDATE inactive_wells SET orig_completion_day = '01' WHERE orig_completion_day = '00' OR orig_completion_day::int > 31
UPDATE inactive_wells SET orig_completion_day = '01' WHERE orig_completion_day = '00' OR SUBSTR(orig_completion, 5, 2) = '00';
UPDATE inactive_wells SET orig_completion = orig_completion_yr || orig_completion_month || orig_completion_day;
UPDATE inactive_wells SET orig_completion = TO_DATE(orig_completion, 'YYYYMMDD');
ALTER TABLE inactive_wells ALTER COLUMN orig_completion TYPE date USING orig_completion::date;

-- Convert orphan list dates
UPDATE orphans SET ice_inspection_date = ice_inspection_date::date WHERE ice_inspection_date <> '';
UPDATE orphans SET ice_inspection_date = NULL WHERE ice_inspection_date = '';
ALTER TABLE orphans ALTER COLUMN ice_inspection_date TYPE date USING ice_inspection_date::date;


-- Cleaning lease/operator names in inactive_wells data
UPDATE inactive_wells SET lease_name = SUBSTR(lease_name, 2, LENGTH(lease_name) - 2) WHERE LEFT(lease_name, 1) = '"' AND RIGHT(lease_name, 1) = '"';
UPDATE inactive_wells SET lease_name = REGEXP_REPLACE(lease_name, '\s+$', '');
UPDATE inactive_wells SET operator_name = SUBSTR(operator_name, 2, LENGTH(operator_name) - 2) WHERE LEFT(operator_name, 1) = '"' AND RIGHT(operator_name, 1) = '"';
UPDATE inactive_wells SET operator_name = REGEXP_REPLACE(operator_name, '\s+$', '');
UPDATE inactive_wells SET field_name = SUBSTR(field_name, 2, LENGTH(field_name) - 2) WHERE LEFT(field_name, 1) = '"' AND RIGHT(field_name, 1) = '"';
UPDATE inactive_wells SET field_name = REGEXP_REPLACE(field_name, '\s+$', '');
UPDATE inactive_wells SET lease_no = LPAD(lease_no, 6, '0') WHERE LENGTH(lease_no) < 6 AND og_code = 'G';
UPDATE inactive_wells SET lease_no = LPAD(lease_no, 5, '0') WHERE LENGTH(lease_no) < 5 AND og_code = 'O';

-- Fix lease numbers in inactive_wells data
UPDATE inactive_wells SET lease_no = LPAD(lease_no, 6, '0') WHERE LENGTH(lease_no) < 6 AND og_code = 'G';
UPDATE inactive_wells SET lease_no = LPAD(lease_no, 5, '0') WHERE LENGTH(lease_no) < 5 AND og_code = 'O';

-- Cleaning operator/field names in orphans data
UPDATE orphans SET operator_name = SUBSTR(operator_name, 2, LENGTH(operator_name) - 2) WHERE LEFT(operator_name, 1) = '"' AND RIGHT(operator_name, 1) = '"';
UPDATE orphans SET operator_name = REGEXP_REPLACE(operator_name, '\s+$', '');
UPDATE orphans SET field_name = SUBSTR(field_name, 2, LENGTH(field_name) - 2) WHERE LEFT(field_name, 1) = '"' AND RIGHT(field_name, 1) = '"';
UPDATE orphans SET field_name = REGEXP_REPLACE(field_name, '\s+$', '');

-- Remove white spaces from well numbers + extra quotes from lease names
UPDATE orphans SET well_no = REPLACE(well_no, ' ', '');
UPDATE inactive_wells SET well_no = REPLACE(well_no, ' ', '');
UPDATE completions SET well_no = REPLACE(well_no, ' ', '');
UPDATE orphans SET lease_name = REPLACE(lease_name, '""', '"') WHERE lease_name LIKE '%""%""%';
UPDATE inactive_wells SET lease_name = REPLACE(lease_name, '""', '"') WHERE lease_name LIKE '%""%""%'; 
UPDATE wellbores SET well_no = REPLACE(well_no, ' ', '');
UPDATE wellbores SET lease_no = REPLACE(lease_no, ' ', '');

-- Adjusting data types
ALTER TABLE orphans ALTER COLUMN bonded_depth TYPE int USING bonded_depth::integer;
ALTER TABLE orphans ALTER COLUMN months_delinquent TYPE int USING months_delinquent::integer;
ALTER TABLE inactive_wells ALTER COLUMN api_depth TYPE int USING api_depth::integer;
ALTER TABLE inactive_wells ALTER COLUMN p5_renewal_month TYPE int USING p5_renewal_month::integer;
ALTER TABLE inactive_wells ALTER COLUMN p5_renewal_year TYPE int USING p5_renewal_year::integer;
ALTER TABLE inactive_wells ALTER COLUMN current_inactive_yrs TYPE int USING current_inactive_yrs::integer;
ALTER TABLE inactive_wells ALTER COLUMN current_inactive_months TYPE int USING current_inactive_months::integer;
ALTER TABLE inactive_wells ALTER COLUMN aged_inactive_yrs TYPE int USING aged_inactive_yrs::integer;
ALTER TABLE inactive_wells ALTER COLUMN aged_inactive_months TYPE int USING aged_inactive_months::integer;
ALTER TABLE inactive_wells ALTER COLUMN cost_calc TYPE int USING cost_calc::integer;

-- Setting blank values to null
UPDATE inactive_wells SET oil_unit_no = NULL WHERE oil_unit_no = ' ';
UPDATE inactive_wells SET current_5yr_inactive = NULL WHERE current_5yr_inactive = ' ';
UPDATE inactive_wells SET current_10yr_inactive = NULL WHERE current_10yr_inactive = ' ';
UPDATE inactive_wells SET aged_5yr_inactive = NULL WHERE aged_5yr_inactive = ' ';
UPDATE inactive_wells SET aged_10yr_inactive = NULL WHERE aged_10yr_inactive = ' ';
UPDATE inactive_wells SET api_depth = NULL WHERE api_depth = 0;
UPDATE inactive_wells SET p5_renewal_month = NULL WHERE p5_renewal_month = 0;
UPDATE inactive_wells SET cost_calc = NULL WHERE cost_calc = 0;
UPDATE orphans SET ofcu_well_priority = NULL WHERE ofcu_well_priority = '';
UPDATE orphans SET sfp_code = NULL WHERE sfp_code = '';
UPDATE orphans SET ice_inspection_id = NULL WHERE ice_inspection_id = '';
UPDATE orphans SET sb639_enf = NULL WHERE sb639_enf = '';
UPDATE orphans SET sb639_r15 = NULL WHERE sb639_r15 = '';
UPDATE orphans SET bonded_depth = NULL WHERE bonded_depth = 0;

-- Set columns that should not be null
ALTER TABLE inactive_wells ALTER COLUMN operator_no SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN operator_name SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN county_name SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN og_code SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN district_code SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN lease_no SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN well_no SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN field_no SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN field_name SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN water_land_code SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN shutin_dt SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN p5_renewal_year SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN p5_originating_status SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN well_plugged SET NOT NULL;
ALTER TABLE inactive_wells ALTER COLUMN api SET NOT NULL;
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
UPDATE inactive_wells SET ext_data_h = 't' WHERE ext_data_h IS NOT NULL;
UPDATE inactive_wells SET ext_data_h = 'f' WHERE ext_data_h IS NULL;
UPDATE inactive_wells SET ext_data_e = 't' WHERE ext_data_e IS NOT NULL;
UPDATE inactive_wells SET ext_data_e = 'f' WHERE ext_data_e IS NULL;
UPDATE inactive_wells SET ext_data_p = 't' WHERE ext_data_p IS NOT NULL;
UPDATE inactive_wells SET ext_data_p = 'f' WHERE ext_data_p IS NULL;
UPDATE inactive_wells SET ext_data_f = 't' WHERE ext_data_f IS NOT NULL;
UPDATE inactive_wells SET ext_data_f = 'f' WHERE ext_data_f IS NULL;
UPDATE inactive_wells SET ext_data_v = 't' WHERE ext_data_v IS NOT NULL;
UPDATE inactive_wells SET ext_data_v = 'f' WHERE ext_data_v IS NULL;
UPDATE inactive_wells SET ext_data_x = 't' WHERE ext_data_x IS NOT NULL;
UPDATE inactive_wells SET ext_data_x = 'f' WHERE ext_data_x IS NULL;
UPDATE inactive_wells SET ext_data_o = 't' WHERE ext_data_o IS NOT NULL;
UPDATE inactive_wells SET ext_data_o = 'f' WHERE ext_data_o IS NULL;
UPDATE inactive_wells SET ext_data_t = 't' WHERE ext_data_t IS NOT NULL;
UPDATE inactive_wells SET ext_data_t = 'f' WHERE ext_data_t IS NULL;
UPDATE inactive_wells SET ext_data_m = 't' WHERE ext_data_m IS NOT NULL;
UPDATE inactive_wells SET ext_data_m = 'f' WHERE ext_data_m IS NULL;
UPDATE inactive_wells SET ext_data_k = 't' WHERE ext_data_k IS NOT NULL;
UPDATE inactive_wells SET ext_data_k = 'f' WHERE ext_data_k IS NULL;
UPDATE inactive_wells SET ext_data_r = 't' WHERE ext_data_r IS NOT NULL;
UPDATE inactive_wells SET ext_data_r = 'f' WHERE ext_data_r IS NULL;
UPDATE inactive_wells SET ext_data_q = 't' WHERE ext_data_q IS NOT NULL;
UPDATE inactive_wells SET ext_data_q = 'f' WHERE ext_data_q IS NULL;
UPDATE inactive_wells SET ext_data_s = 't' WHERE ext_data_s IS NOT NULL;
UPDATE inactive_wells SET ext_data_s = 'f' WHERE ext_data_s IS NULL;
UPDATE inactive_wells SET ext_data_w = 't' WHERE ext_data_w IS NOT NULL;
UPDATE inactive_wells SET ext_data_w = 'f' WHERE ext_data_w IS NULL;
UPDATE inactive_wells SET current_5yr_inactive = 't' WHERE current_5yr_inactive IS NOT NULL;
UPDATE inactive_wells SET current_5yr_inactive = 'f' WHERE current_5yr_inactive IS NULL;
UPDATE inactive_wells SET current_10yr_inactive = 't' WHERE current_10yr_inactive IS NOT NULL;
UPDATE inactive_wells SET current_10yr_inactive = 'f' WHERE current_10yr_inactive IS NULL;
UPDATE inactive_wells SET aged_5yr_inactive = 't' WHERE aged_5yr_inactive IS NOT NULL;
UPDATE inactive_wells SET aged_5yr_inactive = 'f' WHERE aged_5yr_inactive IS NULL;
UPDATE inactive_wells SET aged_10yr_inactive = 't' WHERE aged_10yr_inactive IS NOT NULL;
UPDATE inactive_wells SET aged_10yr_inactive = 'f' WHERE aged_10yr_inactive IS NULL;
UPDATE inactive_wells SET well_plugged = 't' WHERE well_plugged = 'Y';
UPDATE inactive_wells SET well_plugged = 'f' WHERE well_plugged = 'N';
ALTER TABLE inactive_wells ALTER COLUMN ext_data_h TYPE boolean USING ext_data_h::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_e TYPE boolean USING ext_data_e::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_p TYPE boolean USING ext_data_p::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_f TYPE boolean USING ext_data_f::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_v TYPE boolean USING ext_data_v::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_x TYPE boolean USING ext_data_x::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_o TYPE boolean USING ext_data_o::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_t TYPE boolean USING ext_data_t::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_m TYPE boolean USING ext_data_m::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_k TYPE boolean USING ext_data_k::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_r TYPE boolean USING ext_data_r::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_q TYPE boolean USING ext_data_q::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_s TYPE boolean USING ext_data_s::boolean;
ALTER TABLE inactive_wells ALTER COLUMN ext_data_w TYPE boolean USING ext_data_w::boolean;
ALTER TABLE inactive_wells ALTER COLUMN current_5yr_inactive TYPE boolean USING current_5yr_inactive::boolean;
ALTER TABLE inactive_wells ALTER COLUMN current_10yr_inactive TYPE boolean USING current_10yr_inactive::boolean; 
ALTER TABLE inactive_wells ALTER COLUMN aged_5yr_inactive TYPE boolean USING aged_5yr_inactive::boolean; 
ALTER TABLE inactive_wells ALTER COLUMN aged_10yr_inactive TYPE boolean USING aged_10yr_inactive::boolean; 
ALTER TABLE inactive_wells ALTER COLUMN well_plugged TYPE boolean USING well_plugged::boolean;

-- Query to join inactive_wells and orphan wells list based on lease name, lease id, and well number
SELECT inactive_wells.og_code, orphans.operator_name, orphans.operator_no, orphans.lease_name, orphans.lease_no, 
orphans.well_no, orphans.api, orphans.district, orphans.county, orphans.field_name, inactive_wells.field_no, 
inactive_wells.oil_unit_no, inactive_wells.water_land_code, inactive_wells.api_depth, inactive_wells.shutin_dt, inactive_wells.well_plugged, inactive_wells.cost_calc, 
inactive_wells.compliance_due_date,  inactive_wells.orig_completion, orphans.ice_inspection_date, orphans.ice_inspection_id, 
orphans.sfp_code, orphans.ofcu_well_priority, orphans.sb639_enf, orphans.sb639_r15, orphans.months_delinquent, 
inactive_wells.p5_renewal_month, inactive_wells.p5_renewal_year, inactive_wells.p5_originating_status, inactive_wells.current_5yr_inactive, 
inactive_wells.current_10yr_inactive, inactive_wells.aged_5yr_inactive, inactive_wells.aged_10yr_inactive, inactive_wells.current_inactive_yrs, 
inactive_wells.current_inactive_months, inactive_wells.aged_inactive_yrs, inactive_wells.aged_inactive_months, inactive_wells.extension_status, 
inactive_wells.ext_data_h, inactive_wells.ext_data_e, inactive_wells.ext_data_p, inactive_wells.ext_data_f, inactive_wells.ext_data_v, inactive_wells.ext_data_x, 
inactive_wells.ext_data_o, inactive_wells.ext_data_t, inactive_wells.ext_data_m, inactive_wells.ext_data_k, inactive_wells.ext_data_r, inactive_wells.ext_data_q, 
inactive_wells.ext_data_s, inactive_wells.ext_data_w FROM orphans JOIN inactive_wells ON (orphans.lease_name = inactive_wells.lease_name AND 
orphans.lease_no = inactive_wells.lease_no AND orphans.well_no = inactive_wells.well_no) WHERE inactive_wells.orig_completion IS NOT NULL;

-- Query to analyze repeat well API numbers in completions data
/*SELECT * FROM (
	SELECT api, count(*)
	FROM completions
	GROUP BY api
	HAVING count(api) > 1) tbl JOIN completions ON completions.api = tbl.api
	ORDER BY completions.api; */

-- Update cost values where well depth is known but cost is not recorded
UPDATE inactive_wells
	SET cost_calc = CASE
		WHEN district_code = '01' THEN ROUND(11.14 * inactive_wells.api_depth)
		WHEN district_code = '02' THEN ROUND(15.70 * inactive_wells.api_depth)
		WHEN district_code = '03' THEN ROUND(13.93 * inactive_wells.api_depth)
		WHEN district_code = '04' THEN ROUND(8.32 * inactive_wells.api_depth)
		WHEN district_code = '05' THEN ROUND(6.40 * inactive_wells.api_depth)
		WHEN district_code = '06' THEN ROUND(7.93 * inactive_wells.api_depth)
		WHEN district_code = '6E' THEN ROUND(6.79 * inactive_wells.api_depth)
		WHEN district_code = '7B' THEN ROUND(11.27 * inactive_wells.api_depth)
		WHEN district_code = '7C' THEN ROUND(9.63 * inactive_wells.api_depth)
		WHEN district_code = '08' THEN ROUND(17.88 * inactive_wells.api_depth)
		WHEN district_code = '8A' THEN ROUND(12.16 * inactive_wells.api_depth)
		WHEN district_code = '09' THEN ROUND(4.37 * inactive_wells.api_depth)
		WHEN district_code = '10' THEN ROUND(9.69 * inactive_wells.api_depth)
	END
WHERE cost_calc IS NULL AND api_depth IS NOT NULL;

--Feature engineering: price information 
ALTER TABLE inactive_wells ADD COLUMN oil_price_shutin FLOAT;
ALTER TABLE inactive_wells ADD COLUMN gas_price_shutin FLOAT;
ALTER TABLE inactive_wells ADD COLUMN oil_price_shutin_12mo_avg FLOAT;
ALTER TABLE inactive_wells ADD COLUMN gas_price_shutin_12mo_avg FLOAT;

-- Price of oil at well shut in date
UPDATE inactive_wells SET oil_price_shutin = oil_prices.price FROM oil_prices 
WHERE EXTRACT(YEAR FROM oil_prices.date) = EXTRACT(YEAR FROM inactive_wells.shutin_dt) AND
EXTRACT(MONTH FROM oil_prices.date) = EXTRACT(MONTH FROM inactive_wells.shutin_dt);

-- Price of gas at well shut in date
UPDATE inactive_wells SET gas_price_shutin = gas_prices.price FROM gas_prices 
WHERE EXTRACT(YEAR FROM gas_prices.date) = EXTRACT(YEAR FROM inactive_wells.shutin_dt) AND
EXTRACT(MONTH FROM gas_prices.date) = EXTRACT(MONTH FROM inactive_wells.shutin_dt);

-- Average price of oil in the 12 months leading up to the shut in date, including the shut in date
UPDATE inactive_wells
SET oil_price_shutin_12mo_avg = (SELECT AVG(oil_prices.price) FROM oil_prices
								 WHERE oil_prices.date <= inactive_wells.shutin_dt
								 AND oil_prices.date >= inactive_wells.shutin_dt - INTERVAL '1 year');

-- Average price of gas in the 12 months leading up to the shut in date, including the shut in date
UPDATE inactive_wells
SET gas_price_shutin_12mo_avg = (SELECT AVG(gas_prices.price) FROM gas_prices
								 WHERE gas_prices.date <= inactive_wells.shutin_dt
								 AND gas_prices.date >= inactive_wells.shutin_dt - INTERVAL '1 year');

-- Dropped columns create_by and create_dt in operators table (not helpful for us)
ALTER TABLE operators DROP COLUMN create_by;
ALTER TABLE operators DROP COLUMN create_dt;

-- Update production data dates
UPDATE district_prod SET cycle_year_month = cycle_year_month || '15';
UPDATE district_prod SET cycle_year_month = TO_DATE(cycle_year_month, 'YYYYMMDD');
ALTER TABLE district_prod ALTER COLUMN cycle_year_month TYPE date USING cycle_year_month::date;
ALTER TABLE district_prod RENAME COLUMN cycle_year_month TO cycle_date;

UPDATE county_prod SET cycle_year_month = cycle_year_month || '15';
UPDATE county_prod SET cycle_year_month = TO_DATE(cycle_year_month, 'YYYYMMDD');
ALTER TABLE county_prod ALTER COLUMN cycle_year_month TYPE date USING cycle_year_month::date;
ALTER TABLE county_prod RENAME COLUMN cycle_year_month TO cycle_date;

UPDATE operator_prod SET cycle_year_month = cycle_year_month || '15';
UPDATE operator_prod SET cycle_year_month = TO_DATE(cycle_year_month, 'YYYYMMDD');
ALTER TABLE operator_prod ALTER COLUMN cycle_year_month TYPE date USING cycle_year_month::date;
ALTER TABLE operator_prod RENAME COLUMN cycle_year_month TO cycle_date;

-- Making lease names from wellbores/inactive wells/orphans consistent
-- Temporary fix (manually make lease names equal):
UPDATE wellbores 
SET lease_name = orphans.lease_name
FROM orphans
WHERE orphans.api = wellbores.api AND orphans.lease_no = wellbores.lease_no AND orphans.well_no = wellbores.well_no AND orphans.lease_name <> wellbores.lease_name;

-- Query to join orphan wells with all wellbore data
SELECT orphans.api, wellbores.og_code, orphans.district, orphans.county, wellbores.county_no, orphans.ofcu_well_priority, orphans.operator_name, orphans.operator_no,
orphans.lease_name, orphans.lease_no, orphans.well_no, orphans.field_name, wellbores.field_no,
orphans.sfp_code, orphans.ice_inspection_date, orphans.ice_inspection_id, orphans.sb639_enf, orphans.sb639_r15, orphans.months_delinquent,
wellbores.oil_unit_no, inactive_wells.water_land_code, wellbores.multi_comp_flag, wellbores.api_depth, wellbores.wb_shutin_dt, wellbores.well_shutin_dt,
wellbores.wb_14b2_flag,wellbores.well_type_name, wellbores.plug_date, wellbores.plug_lease_name, wellbores.plug_operator_name,
wellbores.recent_permit, wellbores.recent_permit_lease_name, wellbores.recent_permit_operator_no, wellbores.on_schedule,
wellbores.og_wellbore_ewa_id, wellbores.w2g1_filed_date, wellbores.w2g1_date, wellbores.completion_date, wellbores.w3_file_date,
wellbores.p5_renewal_month, wellbores.p5_renewal_year, wellbores.p5_org_status, wellbores.current_inactive_yrs, wellbores.current_inactive_months,
wellbores.wl_14b2_ext_status, wellbores.wl_14b2_mech_integ, wellbores.wl_14b2_plg_ord_sf, wellbores.wl_14b2_pollution, wellbores.wl_14b2_fldops_hold,
wellbores.wl_14b2_h15_prob, wellbores.wl_14b2_h15_delq, wellbores.wl_14b2_oper_delq, wellbores.wl_14b2_dist_sfp, wellbores.wl_14b2_dist_sf_clnup,
wellbores.wl_14b2_dist_st_plg, wellbores.wl_14b2_good_faith, wellbores.wl_14b2_well_other, wellbores.surf_eqp_viol, wellbores.w3x_viol,
wellbores.h15_status_code FROM orphans
JOIN inactive_wells ON (orphans.lease_name = inactive_wells.lease_name AND orphans.lease_no = inactive_wells.lease_no AND
orphans.well_no = inactive_wells.well_no)
JOIN wellbores ON (orphans.lease_name = wellbores.lease_name AND orphans.lease_no = wellbores.lease_no AND orphans.well_no = wellbores.well_no);

-- To view active wells:
SELECT * FROM wellbores WHERE well_type_name = 'PRODUCING' AND wb_shutin_dt IS NULL AND well_shutin_dt = '0';

-- Get inactive wells (not orphans)
SELECT inactive_wells.api, inactive_wells.og_code, inactive_wells.district_code AS district, inactive_wells.county_name, wellbores.county_no,
inactive_wells.operator_name, inactive_wells.operator_no, inactive_wells.lease_name, inactive_wells.lease_no, inactive_wells.well_no,
inactive_wells.field_name, inactive_wells.field_no, inactive_wells.oil_unit_no, inactive_wells.water_land_code, wellbores.multi_comp_flag,
inactive_wells.api_depth, inactive_wells.shutin_dt, inactive_wells.cost_calc, inactive_wells.well_plugged, inactive_wells.compliance_due_date,
wellbores.wb_shutin_dt, wellbores.well_shutin_dt, wellbores.wb_14b2_flag,wellbores.well_type_name, wellbores.plug_date, wellbores.plug_lease_name,
wellbores.plug_operator_name, wellbores.recent_permit, wellbores.recent_permit_lease_name, wellbores.recent_permit_operator_no, wellbores.on_schedule,
wellbores.og_wellbore_ewa_id, wellbores.w2g1_filed_date, wellbores.w2g1_date, wellbores.completion_date, wellbores.w3_file_date, wellbores.p5_renewal_month,
wellbores.p5_renewal_year, wellbores.p5_org_status, wellbores.current_inactive_yrs, wellbores.current_inactive_months, wellbores.wl_14b2_ext_status,
wellbores.wl_14b2_mech_integ, wellbores.wl_14b2_plg_ord_sf, wellbores.wl_14b2_pollution, wellbores.wl_14b2_fldops_hold, wellbores.wl_14b2_h15_prob,
wellbores.wl_14b2_h15_delq, wellbores.wl_14b2_oper_delq, wellbores.wl_14b2_dist_sfp, wellbores.wl_14b2_dist_sf_clnup, wellbores.wl_14b2_dist_st_plg,
wellbores.wl_14b2_good_faith, wellbores.wl_14b2_well_other, wellbores.surf_eqp_viol, wellbores.w3x_viol, wellbores.h15_status_code FROM inactive_wells
LEFT JOIN orphans ON (inactive_wells.lease_name = orphans.lease_name AND inactive_wells.lease_no = orphans.lease_no AND inactive_wells.well_no = orphans.well_no)
JOIN wellbores ON (inactive_wells.api = wellbores.api AND inactive_wells.lease_no = wellbores.lease_no AND inactive_wells.well_no = wellbores.well_no)
WHERE orphans.lease_name IS NULL AND orphans.lease_no IS NULL and orphans.well_no IS NULL;

-- What well types are the inactive wells? Seems to be a mix of types
SELECT DISTINCT well_type_name FROM inactive_wells
JOIN wellbores
ON (inactive_wells.api = wellbores.api AND inactive_wells.lease_no = wellbores.lease_no
	AND inactive_wells.well_no = wellbores.well_no);

-- Update wellbores api numbers
ALTER TABLE wellbores2 ADD COLUMN api VARCHAR;
UPDATE wellbores2 SET api = api_county || api_unique;
ALTER TABLE wellbores2 DROP COLUMN api_county, DROP COLUMN api_unique;

-- Check active wells with available original completion dates
SELECT wellbores2.api, wellbores.district, wellbores.county_no, wellbores.lease_name, wellbores.field_name, wellbores.well_type_name,
wellbores.operator_name, wellbores2.orig_completion, wellbores.well_shutin_dt
FROM wellbores2 JOIN wellbores ON (wellbores2.api = wellbores.api)
WHERE wellbores2.orig_completion <> '00000000' AND wellbores2.orig_completion <> '19840112' AND wellbores.well_type_name = 'PRODUCING'
AND (wellbores.well_shutin_dt = '0' OR wellbores.well_shutin_dt IS NULL);

-- Check inactive wells with available original completion dates
SELECT * FROM inactive_wells
WHERE orig_completion IS NOT NULL AND orig_completion <>'1984-01-12'::date;

-- Check orphans with available original completion dates
SELECT * FROM orphans
JOIN inactive_wells ON (orphans.lease_name = inactive_wells.lease_name AND orphans.lease_no = inactive_wells.lease_no AND orphans.well_no = inactive_wells.well_no)
JOIN wellbores2 ON (orphans.api = wellbores2.api)
WHERE inactive_wells.orig_completion IS NOT NULL AND inactive_wells.orig_completion <> '1984-01-12'::date
AND wellbores2.orig_completion IS NOT NULL AND wellbores2.orig_completion <> '19840112' AND wellbores2.orig_completion <> '00000000';

-- Setting null values across the datasets
 
-- P5 Organizations Data
UPDATE p5_orgs
SET date_built = NULL 
WHERE date_built = '00000000';
UPDATE p5_orgs 
SET organ_other_comment = NULL
WHERE organ_other_comment = '                    ';
UPDATE p5_orgs 
SET gatherer_code = NULL
WHERE gatherer_code = '     ';
UPDATE p5_orgs 
SET renewal_letter_code = NULL
WHERE renewal_letter_code = ' ';
UPDATE p5_orgs 
SET org_address_line1 = NULL
WHERE org_address_line1 LIKE ',%';
UPDATE p5_orgs
SET org_address_line2 = NULL 
WHERE org_address_line2 = '                       -       '
OR org_address_line2 = '                         -     '
OR org_address_line2 = '                           -   '
OR org_address_line2 = '                            -  '
OR org_address_line2 = '                             - '
OR org_address_line2 = '                               ';
UPDATE p5_orgs
SET org_city = NULL
WHERE org_city = '             ';
UPDATE p5_orgs
SET org_state = NULL
WHERE org_state = '  ';
UPDATE p5_orgs
SET location_address_line1 = NULL
WHERE location_address_line1 = '                               ';
UPDATE p5_orgs
SET location_address_line2 = NULL
WHERE location_address_line2 = '                       -       '
OR location_address_line2 = '                         -     '
OR location_address_line2 = '                           -   '
OR location_address_line2 = '                            -  '
OR location_address_line2 = '                             - '
OR location_address_line2 = '                               ';
UPDATE p5_orgs
SET location_address_line2 = NULL
WHERE location_address_line2 = '+                              '
UPDATE p5_orgs
SET location_city = NULL
WHERE location_city = '             ';
UPDATE p5_orgs
SET location_state = NULL
WHERE location_state = '  ';
UPDATE p5_orgs
SET date_inactive = NULL
WHERE date_inactive = '00000000';
UPDATE p5_orgs
SET org_phone_num = NULL
WHERE org_phone_num = '0000000000';
UPDATE p5_orgs
SET refile_notice_month = NULL
WHERE refile_notice_month = '00';
UPDATE p5_orgs
SET refile_letter_date = NULL
WHERE refile_letter_date = '00000000';
UPDATE p5_orgs
SET refile_notice_date = NULL
WHERE refile_notice_date = '00000000';
UPDATE p5_orgs
SET refile_received_date = NULL
WHERE refile_received_date = '00000000';
UPDATE p5_orgs
SET last_p5_received_date = NULL
WHERE last_p5_received_date = '19000000';
UPDATE p5_orgs
SET other_org_no = NULL
WHERE other_org_no = '000000';
UPDATE p5_orgs
SET filing_problem_date = NULL
WHERE filing_problem_date = '00000000'
OR filing_problem_date = '        ';
UPDATE p5_orgs
SET filing_problem_ltr_code = NULL
WHERE filing_problem_ltr_code = '   ';
UPDATE p5_orgs
SET op_num_multi_used_flag = 'N'
WHERE op_num_multi_used_flag = ' ';
UPDATE p5_orgs
SET oil_gatherer_status = 'N'
WHERE oil_gatherer_status = ' ';
UPDATE p5_orgs
SET gas_gatherer_status = 'N'
WHERE gas_gatherer_status = ' ';
UPDATE p5_orgs
SET tax_cert = 'NR' -- NR stands for Not Requested here
WHERE tax_cert = ' ';
UPDATE p5_orgs
SET emergency_phone_num = NULL
WHERE emergency_phone_num = '0000000000';

-- Wellbores2 (second wellbores dataset)
UPDATE wellbores2
SET api = NULL
WHERE api = 'x'
OR api = 'UNKNOWN'
OR api = 'Unknown'
OR api = 'unknown'
OR api = 'UNK'
OR api = 'unk'
OR api = 'UKN'
OR api = 'N/AQ'
OR api = 'na';

UPDATE wellbores2 SET orig_completion = NULL WHERE orig_completion = '00000000';
UPDATE wellbores2 SET orig_completion = TO_DATE(orig_completion, 'YYYYMMDD');
ALTER TABLE wellbores2 ALTER COLUMN orig_completion TYPE date USING orig_completion::date;
-- Weird dates
UPDATE wellbores2 SET orig_completion = '19430930' WHERE orig_completion = '19430931';
UPDATE wellbores2 SET orig_completion = '19770430' WHERE orig_completion = '19770431';
UPDATE wellbores2 SET orig_completion = '19550228' WHERE orig_completion = '19550229';