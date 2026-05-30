
select
 min(i_item_id)
 ,min(i_item_desc)
 ,min(s_store_id)
 ,min(s_store_name)
 ,min(ss_net_profit)
 ,min(sr_net_loss)
 ,min(cs_net_profit)
 ,min(ss_item_sk)
 ,min(sr_ticket_number)
 ,min(cs_order_number)
 from
 store_sales
 ,store_returns
 ,catalog_sales
 ,date_dim date_dim1
 ,date_dim date_dim2
 ,date_dim date_dim3
 ,store
 ,item
 where
 date_dim1.d_moy = 1
 and date_dim1.d_year = 2000
 and date_dim1.d_date_sk = ss_sold_date_sk
 and i_item_sk = ss_item_sk
 and s_store_sk = ss_store_sk
 and ss_customer_sk = sr_customer_sk
 and ss_item_sk = sr_item_sk
 and ss_ticket_number = sr_ticket_number
 and sr_returned_date_sk = date_dim2.d_date_sk
 and date_dim2.d_moy               between 1 and  1 + 2
 and date_dim2.d_year              = 2000
 and sr_customer_sk = cs_bill_customer_sk
 and sr_item_sk = cs_item_sk
 and cs_sold_date_sk = date_dim3.d_date_sk
 and date_dim3.d_moy               between 1 and  1 + 2
 and date_dim3.d_year              = 2000
 ;


