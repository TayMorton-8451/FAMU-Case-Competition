# Databricks notebook source
######PROJECT SETUP NECESSARY
### Import Libraries
import pandas as pd
import numpy as np
from effodata import ACDS, golden_rules, Joiner, Sifter, Equality
from kpi_metrics import KPI, AliasMetric, CustomMetric, AliasGroupby
import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql.window import Window
import re
from pyspark.sql.types import StringType 
from shutil import copyfile
from IPython.display import FileLink 
import os
import sys
import time
import seg
from seg.utils import DateType
import upc_input

# COMMAND ----------

#######LOADING ACDS AS A SAMPLE CURRENTLY
## ALSO LOADING IN KPI
## CHANGE THIS TO FALSE WHEN WE'RE FINALLY USING IT
acds = ACDS(use_sample_mart = False)
kpi = KPI(use_sample_mart = False)

# COMMAND ----------

## Import in Big Pack UPCs
# File location and type
file_location = "/FileStore/tables/m434100/Big_Pack_UPCs.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
big_upcs = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)


# Create a view or table to join to later with SQL
big_upcs.createOrReplaceTempView("big_upcs")
display(big_upcs)

# COMMAND ----------

## Need Divs for Fred Meyer, Columbus, Frys, Michigan, Kings
## divs = acds.stores.select('mgt_div_no','mgt_div_dsc_tx').distinct()
## display(divs)
## 016, 660, 701, 620, 018

# COMMAND ----------

big_pack_UPCs_pre = acds.get_transactions(\
           start_date="20190310",\
           end_date="20200307",\
           join_with=["stores", "products"],\
           apply_golden_rules=golden_rules(),\
           query_filters=["mgt_div_no in ('016','660','701','620','018')","pid_fyt_com_cd in ('024','055','065','075','112','180','205','435')"]).\
           select('trn_dt','gtin_no','con_dsc_tx','con_siz_tx','net_spend_amt','scn_unt_qy','transaction_code','ehhn','pid_fyt_com_cd','pid_fyt_com_dsc_tx','order_type')

big_pack_UPCs_covid = acds.get_transactions(\
           start_date="20200308",\
           end_date="20210306",\
           join_with=["stores", "products"],\
           apply_golden_rules=golden_rules(),\
           query_filters=["mgt_div_no in ('016','660','701','620','018')","pid_fyt_com_cd in ('024','055','065','075','112','180','205','435')"]).\
           select('trn_dt','gtin_no','con_dsc_tx','con_siz_tx','net_spend_amt','scn_unt_qy','transaction_code','ehhn','pid_fyt_com_cd','pid_fyt_com_dsc_tx','order_type')

# COMMAND ----------

## Get HH denominator for same time period and divs for overall HH pen
HH_denom_pre = acds.get_transactions(\
           start_date="20190310",\
           end_date="20200307",\
           join_with=["stores"],\
           apply_golden_rules=golden_rules(),\
           query_filters=["mgt_div_no in ('016','660','701','620','018')"]).\
           select('ehhn').agg(f.countDistinct("ehhn").alias("tot_hhs"))

HH_denom_pre = HH_denom_pre.collect()[0][0]

HH_denom_covid = acds.get_transactions(\
           start_date="20200308",\
           end_date="20210306",\
           join_with=["stores"],\
           apply_golden_rules=golden_rules(),\
           query_filters=["mgt_div_no in ('016','660','701','620','018')"]).\
           select('ehhn').agg(f.countDistinct("ehhn").alias("tot_hhs"))

HH_denom_covid = HH_denom_covid.collect()[0][0]

# COMMAND ----------

# Gather stats to assess what kind of match type and parameters you should set in review_upcs()
upc_input.gather_match_stats(
    kpi=kpi,
    df=big_upcs,         # On an input dataframe named 'my_df'...
    upc_col='Consumer' # Which has a upc column named 'my_upcs'
)

#Return three dataframes for continuing your project
all_the_upcs = upc_input.review_upcs(
    kpi=kpi,
    upc_df = big_upcs,           # an input dataframe named 'my_df' ...
    match_type = 'scan',      # where the upcs are 'scan' upcs ...
    upc_col = 'Consumer',      # in a column named 'my_upcs' ...
    start_date = '20190310',  # pulling metrics starting Aug 1, 2019 ...
    end_date = '20210306'     # through Sept 1, 2019
)

# Access the dataframes you generated
joined, metrics_by_upc, metrics_by_agg_level = [df.toPandas()
                                                for df in all_the_upcs]

# COMMAND ----------

sparkDF=spark.createDataFrame(joined) 
big_upcs_filter = sparkDF.select('con_upc_no').distinct()
##display(big_upcs_filter)

# COMMAND ----------

pre_w_segs = seg.get_segs_and_join(["funlo","cds_hh"],"20200307",big_pack_UPCs_pre)\
                .select('trn_dt','gtin_no','con_dsc_tx','net_spend_amt','scn_unt_qy','transaction_code','ehhn','pid_fyt_com_cd','pid_fyt_com_dsc_tx','order_type','funlo_rollup_desc','price_dim_seg')\
                .join(big_upcs_filter, (big_pack_UPCs_pre.gtin_no == big_upcs_filter.con_upc_no), "left")\
                .withColumn("Big_Pack_UPC", f.when(f.col("con_upc_no").isNotNull(), "Y").otherwise("N"))

covid_w_segs = seg.get_segs_and_join(["funlo","cds_hh"],"20210306",big_pack_UPCs_covid)\
                .select('trn_dt','gtin_no','con_dsc_tx','net_spend_amt','scn_unt_qy','transaction_code','ehhn','pid_fyt_com_cd','pid_fyt_com_dsc_tx','order_type','funlo_rollup_desc','price_dim_seg')\
                .join(big_upcs_filter, (big_pack_UPCs_covid.gtin_no == big_upcs_filter.con_upc_no), "left")\
                .withColumn("Big_Pack_UPC", f.when(f.col("con_upc_no").isNotNull(), "Y").otherwise("N"))

# COMMAND ----------

## Left join for these UPCs to Flag as Big Pack
pre_with_flag = big_pack_UPCs_pre.join(big_upcs_filter, (big_pack_UPCs_pre.gtin_no == big_upcs_filter.con_upc_no), "left")\
                                 .withColumn("Big_Pack_UPC", f.when(f.col("con_upc_no").isNotNull(), "Y").otherwise("N"))

covid_with_flag = big_pack_UPCs_covid.join(big_upcs_filter, (big_pack_UPCs_covid.gtin_no == big_upcs_filter.con_upc_no), "left")\
                                 .withColumn("Big_Pack_UPC", f.when(f.col("con_upc_no").isNotNull(), "Y").otherwise("N"))

# COMMAND ----------

##Find top50 non Big Pack  (dense rank on Sales in each commodity)
pre_sales = pre_with_flag.select('gtin_no','pid_fyt_com_cd','net_spend_amt').filter(f.col("Big_Pack_UPC")=="N").groupby(f.col("gtin_no")).agg(f.sum(f.col("net_spend_amt")).alias("sales"))\
                         .join(pre_with_flag.select('gtin_no','pid_fyt_com_cd').distinct(), "gtin_no", "inner")

windowSpec = Window.partitionBy("pid_fyt_com_cd").orderBy(f.col("sales").desc())
pre_ranking = pre_sales.withColumn("upc_sales_rank", f.dense_rank().over(windowSpec))
top_50_pre = pre_ranking.filter(f.col("upc_sales_rank")<=50).join(big_pack_UPCs_pre,"gtin_no","inner")

covid_sales = covid_with_flag.select('gtin_no','pid_fyt_com_cd','net_spend_amt').filter(f.col("Big_Pack_UPC")=="N").groupby(f.col("gtin_no")).agg(f.sum(f.col("net_spend_amt")).alias("sales"))\
                         .join(covid_with_flag.select('gtin_no','pid_fyt_com_cd').distinct(), "gtin_no", "inner")

windowSpec = Window.partitionBy("pid_fyt_com_cd").orderBy(f.col("sales").desc())
covid_ranking = covid_sales.withColumn("upc_sales_rank", f.dense_rank().over(windowSpec))
top_50_covid = covid_ranking.filter(f.col("upc_sales_rank")<=50).join(big_pack_UPCs_covid,"gtin_no","inner")

# COMMAND ----------

##top_50_covid.limit(10).toPandas()
pre_w_segs.limit(10).toPandas()

# COMMAND ----------

## NonBIGS: Get pre & post sales, HHs, units, trips, HH Pen, $/HH, Trips/HH, $/Trip, Units/Trip, $/Unit, Repeat Rate Overall (HHs with Visits > 1 / HHs)
pre_mets_HH = top_50_pre.groupBy("ehhn","gtin_no")\
                        .agg(f.sum(f.col("net_spend_amt")).alias("pre_sales"),\
                             f.sum(f.col("scn_unt_qy")).alias("pre_units"),\
                             f.countDistinct(f.col("transaction_code")).alias("pre_visits"))

covid_mets_HH = top_50_covid.groupBy("ehhn","gtin_no")\
                        .agg(f.sum(f.col("net_spend_amt")).alias("covid_sales"),\
                             f.sum(f.col("scn_unt_qy")).alias("covid_units"),\
                             f.countDistinct(f.col("transaction_code")).alias("covid_visits"))

# COMMAND ----------

pre_repeat_mets_HH = pre_mets_HH.filter(f.col("pre_visits")>1).groupBy("gtin_no").agg(f.countDistinct("ehhn").alias("Pre_Repeat_HHs"))

covid_repeat_mets_HH = covid_mets_HH.filter(f.col("covid_visits")>1).groupBy("gtin_no").agg(f.countDistinct("ehhn").alias("Covid_Repeat_HHs"))

# COMMAND ----------

pre_mets = pre_mets_HH.groupBy("gtin_no").agg(f.sum(f.col("pre_sales")).alias("Pre_Total_Sales"),\
                           f.sum(f.col("pre_units")).alias("Pre_Total_Units"),\
                           f.sum(f.col("pre_visits")).alias("Pre_Total_Visits"),\
                           f.countDistinct(f.col("ehhn")).alias("Pre_Total_HHs"))\
                       .withColumn("Pre_HH_Pen", (f.col("Pre_Total_HHs")/HH_denom_pre)*100)\
                       .withColumn("Pre_Spend_per_HH", f.col("Pre_Total_Sales")/f.col("Pre_Total_HHs"))\
                       .withColumn("Pre_Visits_per_HH", f.col("Pre_Total_Visits")/f.col("Pre_Total_HHs"))\
                       .withColumn("Pre_Spend_per_Visit", f.col("Pre_Total_Sales")/f.col("Pre_Total_Visits"))\
                       .withColumn("Pre_Units_per_Visit", f.col("Pre_Total_Units")/f.col("Pre_Total_Visits"))\
                       .withColumn("Pre_Spend_per_Unit", f.col("Pre_Total_Sales")/f.col("Pre_Total_Units"))\
                       .join(pre_repeat_mets_HH, "gtin_no", "full")\
                       .withColumn("Pre_Repeat_Rate", (f.col("Pre_Repeat_HHs")/f.col("Pre_Total_HHs"))*100)

covid_mets = covid_mets_HH.groupBy("gtin_no").agg(f.sum(f.col("covid_sales")).alias("Covid_Total_Sales"),\
                           f.sum(f.col("covid_units")).alias("Covid_Total_Units"),\
                           f.sum(f.col("covid_visits")).alias("Covid_Total_Visits"),\
                           f.countDistinct(f.col("ehhn")).alias("Covid_Total_HHs"))\
                       .withColumn("Covid_HH_Pen", (f.col("Covid_Total_HHs")/HH_denom_covid)*100)\
                       .withColumn("Covid_Spend_per_HH", f.col("Covid_Total_Sales")/f.col("Covid_Total_HHs"))\
                       .withColumn("Covid_Visits_per_HH", f.col("Covid_Total_Visits")/f.col("Covid_Total_HHs"))\
                       .withColumn("Covid_Spend_per_Visit", f.col("Covid_Total_Sales")/f.col("Covid_Total_Visits"))\
                       .withColumn("Covid_Units_per_Visit", f.col("Covid_Total_Units")/f.col("Covid_Total_Visits"))\
                       .withColumn("Covid_Spend_per_Unit", f.col("Covid_Total_Sales")/f.col("Covid_Total_Units"))\
                       .join(covid_repeat_mets_HH, "gtin_no", "full")\
                       .withColumn("Covid_Repeat_Rate", (f.col("Covid_Repeat_HHs")/f.col("Covid_Total_HHs"))*100)

tot_mets = pre_mets.join(covid_mets, "gtin_no", "full")

# COMMAND ----------

## Get pre & post loyal, PS sales
pre_seg_loy = pre_w_segs.filter(f.col("Big_Pack_UPC")=="N")\
                        .join(top_50_pre.select('gtin_no').distinct(), "gtin_no", "inner")\
                        .filter(f.col("funlo_rollup_desc")=="Loyal")\
                        .groupby('gtin_no')\
                        .agg(f.sum("net_spend_amt").alias("Pre_Loyal_Sales"))

pre_seg_price = pre_w_segs.filter(f.col("Big_Pack_UPC")=="N")\
                        .join(top_50_pre.select('gtin_no').distinct(), "gtin_no", "inner")\
                        .filter(f.col("price_dim_seg")=="H")\
                        .groupby('gtin_no')\
                        .agg(f.sum("net_spend_amt").alias("Pre_Price_Sens_Sales"))

pre_segs = pre_seg_loy.join(pre_seg_price, "gtin_no", "full")

pre_segs.limit(10).toPandas()

covid_seg_loy = covid_w_segs.filter(f.col("Big_Pack_UPC")=="N")\
                        .join(top_50_covid.select('gtin_no').distinct(), "gtin_no", "inner")\
                        .filter(f.col("funlo_rollup_desc")=="Loyal")\
                        .groupby('gtin_no')\
                        .agg(f.sum("net_spend_amt").alias("Covid_Loyal_Sales"))

covid_seg_price = covid_w_segs.filter(f.col("Big_Pack_UPC")=="N")\
                        .join(top_50_covid.select('gtin_no').distinct(), "gtin_no", "inner")\
                        .filter(f.col("price_dim_seg")=="H")\
                        .groupby('gtin_no')\
                        .agg(f.sum("net_spend_amt").alias("Covid_Price_Sens_Sales"))

covid_segs = covid_seg_loy.join(covid_seg_price, "gtin_no", "full")

tot_segs = pre_segs.join(covid_segs, "gtin_no", "full")

## Get pre & post Pickup sales, HHs, trips, Units
pre_ecomm = top_50_pre.filter(f.col("order_type")!= "")\
                      .groupby("gtin_no")\
                      .agg(f.sum(f.col("net_spend_amt")).alias("Pre_Ecomm_Sales"),\
                           f.countDistinct(f.col("ehhn")).alias("Pre_Ecomm_HHs"),\
                           f.countDistinct(f.col("transaction_code")).alias("Pre_Ecomm_Visits"),\
                           f.sum(f.col("scn_unt_qy")).alias("Pre_Ecomm_Units"))

covid_ecomm= top_50_covid.filter(f.col("order_type")!= "")\
                      .groupby("gtin_no")\
                      .agg(f.sum(f.col("net_spend_amt")).alias("Covid_Ecomm_Sales"),\
                           f.countDistinct(f.col("ehhn")).alias("Covid_Ecomm_HHs"),\
                           f.countDistinct(f.col("transaction_code")).alias("Covid_Ecomm_Visits"),\
                           f.sum(f.col("scn_unt_qy")).alias("Covid_Ecomm_Units"))

tot_ecomm = pre_ecomm.join(covid_ecomm, "gtin_no", "full")

# COMMAND ----------

top_50_all = tot_mets.join(tot_segs, "gtin_no", "full").join(tot_ecomm, "gtin_no", "full")

# COMMAND ----------

display(top_50_all)

# COMMAND ----------

## Filter to Big Pack Items & Repeat the above metrics
big_pre_sales = pre_with_flag.select('gtin_no','pid_fyt_com_cd','net_spend_amt').filter(f.col("Big_Pack_UPC")=="Y").groupby(f.col("gtin_no")).agg(f.sum(f.col("net_spend_amt")).alias("sales"))\
                         .join(pre_with_flag.select('gtin_no','pid_fyt_com_cd').distinct(), "gtin_no", "inner")

windowSpec = Window.partitionBy("pid_fyt_com_cd").orderBy(f.col("sales").desc())
big_pre_ranking = big_pre_sales.withColumn("upc_sales_rank", f.dense_rank().over(windowSpec))
big_top_50_pre = big_pre_ranking.join(big_pack_UPCs_pre,"gtin_no","inner")

big_covid_sales = covid_with_flag.select('gtin_no','pid_fyt_com_cd','net_spend_amt').filter(f.col("Big_Pack_UPC")=="Y").groupby(f.col("gtin_no")).agg(f.sum(f.col("net_spend_amt")).alias("sales"))\
                         .join(covid_with_flag.select('gtin_no','pid_fyt_com_cd').distinct(), "gtin_no", "inner")

windowSpec = Window.partitionBy("pid_fyt_com_cd").orderBy(f.col("sales").desc())
big_covid_ranking = big_covid_sales.withColumn("upc_sales_rank", f.dense_rank().over(windowSpec))
big_top_50_covid = big_covid_ranking.join(big_pack_UPCs_covid,"gtin_no","inner")

# COMMAND ----------

## BIGS: Get pre & post sales, HHs, units, trips, HH Pen, $/HH, Trips/HH, $/Trip, Units/Trip, $/Unit, Repeat Rate Overall (HHs with Visits > 1 / HHs)
big_pre_mets_HH = big_top_50_pre.groupBy("ehhn","gtin_no")\
                        .agg(f.sum(f.col("net_spend_amt")).alias("pre_sales"),\
                             f.sum(f.col("scn_unt_qy")).alias("pre_units"),\
                             f.countDistinct(f.col("transaction_code")).alias("pre_visits"))

big_covid_mets_HH = big_top_50_covid.groupBy("ehhn","gtin_no")\
                        .agg(f.sum(f.col("net_spend_amt")).alias("covid_sales"),\
                             f.sum(f.col("scn_unt_qy")).alias("covid_units"),\
                             f.countDistinct(f.col("transaction_code")).alias("covid_visits"))

# COMMAND ----------

big_pre_repeat_mets_HH = big_pre_mets_HH.filter(f.col("pre_visits")>1).groupBy("gtin_no").agg(f.countDistinct("ehhn").alias("Pre_Repeat_HHs"))

big_covid_repeat_mets_HH = big_covid_mets_HH.filter(f.col("covid_visits")>1).groupBy("gtin_no").agg(f.countDistinct("ehhn").alias("Covid_Repeat_HHs"))

# COMMAND ----------

big_pre_mets = big_pre_mets_HH.groupBy("gtin_no").agg(f.sum(f.col("pre_sales")).alias("Pre_Total_Sales"),\
                           f.sum(f.col("pre_units")).alias("Pre_Total_Units"),\
                           f.sum(f.col("pre_visits")).alias("Pre_Total_Visits"),\
                           f.countDistinct(f.col("ehhn")).alias("Pre_Total_HHs"))\
                       .withColumn("Pre_HH_Pen", (f.col("Pre_Total_HHs")/HH_denom_pre)*100)\
                       .withColumn("Pre_Spend_per_HH", f.col("Pre_Total_Sales")/f.col("Pre_Total_HHs"))\
                       .withColumn("Pre_Visits_per_HH", f.col("Pre_Total_Visits")/f.col("Pre_Total_HHs"))\
                       .withColumn("Pre_Spend_per_Visit", f.col("Pre_Total_Sales")/f.col("Pre_Total_Visits"))\
                       .withColumn("Pre_Units_per_Visit", f.col("Pre_Total_Units")/f.col("Pre_Total_Visits"))\
                       .withColumn("Pre_Spend_per_Unit", f.col("Pre_Total_Sales")/f.col("Pre_Total_Units"))\
                       .join(big_pre_repeat_mets_HH, "gtin_no", "full")\
                       .withColumn("Pre_Repeat_Rate", (f.col("Pre_Repeat_HHs")/f.col("Pre_Total_HHs"))*100)

big_covid_mets = big_covid_mets_HH.groupBy("gtin_no").agg(f.sum(f.col("covid_sales")).alias("Covid_Total_Sales"),\
                           f.sum(f.col("covid_units")).alias("Covid_Total_Units"),\
                           f.sum(f.col("covid_visits")).alias("Covid_Total_Visits"),\
                           f.countDistinct(f.col("ehhn")).alias("Covid_Total_HHs"))\
                       .withColumn("Covid_HH_Pen", (f.col("Covid_Total_HHs")/HH_denom_covid)*100)\
                       .withColumn("Covid_Spend_per_HH", f.col("Covid_Total_Sales")/f.col("Covid_Total_HHs"))\
                       .withColumn("Covid_Visits_per_HH", f.col("Covid_Total_Visits")/f.col("Covid_Total_HHs"))\
                       .withColumn("Covid_Spend_per_Visit", f.col("Covid_Total_Sales")/f.col("Covid_Total_Visits"))\
                       .withColumn("Covid_Units_per_Visit", f.col("Covid_Total_Units")/f.col("Covid_Total_Visits"))\
                       .withColumn("Covid_Spend_per_Unit", f.col("Covid_Total_Sales")/f.col("Covid_Total_Units"))\
                       .join(big_covid_repeat_mets_HH, "gtin_no", "full")\
                       .withColumn("Covid_Repeat_Rate", (f.col("Covid_Repeat_HHs")/f.col("Covid_Total_HHs"))*100)

big_tot_mets = big_pre_mets.join(big_covid_mets, "gtin_no", "full")

# COMMAND ----------

## Get pre & post loyal, PS sales
big_pre_seg_loy = pre_w_segs.filter(f.col("Big_Pack_UPC")=="Y")\
                        .join(big_top_50_pre.select('gtin_no').distinct(), "gtin_no", "inner")\
                        .filter(f.col("funlo_rollup_desc")=="Loyal")\
                        .groupby('gtin_no')\
                        .agg(f.sum("net_spend_amt").alias("Pre_Loyal_Sales"))

big_pre_seg_price = pre_w_segs.filter(f.col("Big_Pack_UPC")=="Y")\
                        .join(top_50_pre.select('gtin_no').distinct(), "gtin_no", "inner")\
                        .filter(f.col("price_dim_seg")=="H")\
                        .groupby('gtin_no')\
                        .agg(f.sum("net_spend_amt").alias("Pre_Price_Sens_Sales"))

big_pre_segs = big_pre_seg_loy.join(big_pre_seg_price, "gtin_no", "full")

big_covid_seg_loy = covid_w_segs.filter(f.col("Big_Pack_UPC")=="Y")\
                        .join(big_top_50_covid.select('gtin_no').distinct(), "gtin_no", "inner")\
                        .filter(f.col("funlo_rollup_desc")=="Loyal")\
                        .groupby('gtin_no')\
                        .agg(f.sum("net_spend_amt").alias("Covid_Loyal_Sales"))

big_covid_seg_price = covid_w_segs.filter(f.col("Big_Pack_UPC")=="Y")\
                        .join(big_top_50_covid.select('gtin_no').distinct(), "gtin_no", "inner")\
                        .filter(f.col("price_dim_seg")=="H")\
                        .groupby('gtin_no')\
                        .agg(f.sum("net_spend_amt").alias("Covid_Price_Sens_Sales"))

big_covid_segs = big_covid_seg_loy.join(big_covid_seg_price, "gtin_no", "full")

big_tot_segs = big_pre_segs.join(big_covid_segs, "gtin_no", "full")

## Get pre & post Pickup sales, HHs, trips, Units
big_pre_ecomm = big_top_50_pre.filter(f.col("order_type")!= "")\
                      .groupby("gtin_no")\
                      .agg(f.sum(f.col("net_spend_amt")).alias("Pre_Ecomm_Sales"),\
                           f.countDistinct(f.col("ehhn")).alias("Pre_Ecomm_HHs"),\
                           f.countDistinct(f.col("transaction_code")).alias("Pre_Ecomm_Visits"),\
                           f.sum(f.col("scn_unt_qy")).alias("Pre_Ecomm_Units"))

big_covid_ecomm= big_top_50_covid.filter(f.col("order_type")!= "")\
                      .groupby("gtin_no")\
                      .agg(f.sum(f.col("net_spend_amt")).alias("Covid_Ecomm_Sales"),\
                           f.countDistinct(f.col("ehhn")).alias("Covid_Ecomm_HHs"),\
                           f.countDistinct(f.col("transaction_code")).alias("Covid_Ecomm_Visits"),\
                           f.sum(f.col("scn_unt_qy")).alias("Covid_Ecomm_Units"))

big_tot_ecomm = big_pre_ecomm.join(big_covid_ecomm, "gtin_no", "full")

# COMMAND ----------

big_all = big_tot_mets.join(big_tot_segs, "gtin_no", "full").join(big_tot_ecomm, "gtin_no", "full")

# COMMAND ----------

##display(big_all)

# COMMAND ----------

## Join Big Pack Mets & Non Big Pack Mets & Sort by Commodity
all_prods = top_50_all.union(big_all)
## Add sizes of products / description / commodity
## Get list of hierarchy, prod size, descriptor
prods_pre = big_pack_UPCs_pre.select('gtin_no','con_dsc_tx','con_siz_tx','pid_fyt_com_dsc_tx').distinct()
prods_covid = big_pack_UPCs_covid.select('gtin_no','con_dsc_tx','con_siz_tx','pid_fyt_com_dsc_tx').distinct()
prod_descs = prods_pre.union(prods_covid).distinct().join(big_upcs_filter, (prods_pre.gtin_no == big_upcs_filter.con_upc_no), "left")\
                                 .withColumn("Big_Pack_UPC", f.when(f.col("con_upc_no").isNotNull(), "Y").otherwise("N")).drop('con_upc_no')


final_df_upcs = prod_descs.join(all_prods, "gtin_no", "inner")\
                                 .withColumnRenamed("gtin_no","UPC")\
                                 .withColumnRenamed("con_dsc_tx","Description")\
                                 .withColumnRenamed("con_siz_tx","Size")\
                                 .withColumnRenamed("pid_fyt_com_dsc_tx","Category")
##display(final_df_upcs)

# COMMAND ----------

## Write to CSV in ADLS and email to Sarah & Hayley
final_df_upcs.repartition(1).write.mode('overwrite').csv('abfss://users@sa8451dbxadhocprd.dfs.core.windows.net/m434100/UC_case_big_pack_mets_fin',header=True)

# COMMAND ----------


