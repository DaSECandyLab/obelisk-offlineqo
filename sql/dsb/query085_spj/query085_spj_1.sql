
select min(ws_quantity)
       ,min(wr_refunded_cash)
       ,min(wr_fee)
       ,min(ws_item_sk)
       ,min(wr_order_number)
       ,min(customer_demographics1.cd_demo_sk)
	   ,min(customer_demographics2.cd_demo_sk)
 from web_sales, web_returns, web_page, customer_demographics customer_demographics1,
      customer_demographics customer_demographics2, customer_address, date_dim, reason
 where ws_web_page_sk = wp_web_page_sk
   and ws_item_sk = wr_item_sk
   and ws_order_number = wr_order_number
   and ws_sold_date_sk = d_date_sk and d_year = 1998
   and customer_demographics1.cd_demo_sk = wr_refunded_cdemo_sk
   and customer_demographics2.cd_demo_sk = wr_returning_cdemo_sk
   and ca_address_sk = wr_refunded_addr_sk
   and r_reason_sk = wr_reason_sk
   and
   (
    (
     customer_demographics1.cd_marital_status = 'M'
     and
     customer_demographics1.cd_marital_status = customer_demographics2.cd_marital_status
     and
     customer_demographics1.cd_education_status = '2 yr Degree'
     and
     customer_demographics1.cd_education_status = customer_demographics2.cd_education_status
     and
     ws_sales_price between 100.00 and 150.00
    )
   or
    (
     customer_demographics1.cd_marital_status = 'S'
     and
     customer_demographics1.cd_marital_status = customer_demographics2.cd_marital_status
     and
     customer_demographics1.cd_education_status = 'Unknown'
     and
     customer_demographics1.cd_education_status = customer_demographics2.cd_education_status
     and
     ws_sales_price between 50.00 and 100.00
    )
   or
    (
     customer_demographics1.cd_marital_status = 'D'
     and
     customer_demographics1.cd_marital_status = customer_demographics2.cd_marital_status
     and
     customer_demographics1.cd_education_status = 'Advanced Degree'
     and
     customer_demographics1.cd_education_status = customer_demographics2.cd_education_status
     and
     ws_sales_price between 150.00 and 200.00
    )
   )
   and
   (
    (
     ca_country = 'United States'
     and
     ca_state in ('MN', 'OK', 'WV')
     and ws_net_profit between 100 and 200
    )
    or
    (
     ca_country = 'United States'
     and
     ca_state in ('IL', 'OK', 'VA')
     and ws_net_profit between 150 and 300
    )
    or
    (
     ca_country = 'United States'
     and
     ca_state in ('GA', 'KY', 'VA')
     and ws_net_profit between 50 and 250
    )
   )
 ;


