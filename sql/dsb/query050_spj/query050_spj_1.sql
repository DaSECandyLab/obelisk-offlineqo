
select
   min(s_store_name)
  ,min(s_company_id)
  ,min(s_street_number)
  ,min(s_street_name)
  ,min(s_suite_number)
  ,min(s_city)
  ,min(s_zip)
  ,min(ss_ticket_number)
  ,min(ss_sold_date_sk)
  ,min(sr_returned_date_sk)
  ,min(ss_item_sk)
  ,min(date_dim1.d_date_sk)
from
   store_sales
  ,store_returns
  ,store
  ,date_dim date_dim1
  ,date_dim date_dim2
where
    date_dim2.d_moy = 1
and ss_ticket_number = sr_ticket_number
and ss_item_sk = sr_item_sk
and ss_sold_date_sk   = date_dim1.d_date_sk
and sr_returned_date_sk   = date_dim2.d_date_sk
and ss_customer_sk = sr_customer_sk
and ss_store_sk = s_store_sk
and sr_store_sk = s_store_sk
and date_dim1.d_date between DATE_SUB(date_dim2.d_date, interval 120 day)
               and date_dim2.d_date
and date_dim1.d_dow = 4
and s_state in ('KS','LA','OK')
 ;


