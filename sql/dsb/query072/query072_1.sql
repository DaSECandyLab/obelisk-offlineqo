
select  i_item_desc
      ,w_warehouse_name
      ,date_dim1.d_week_seq
      ,sum(case when p_promo_sk is null then 1 else 0 end) no_promo
      ,sum(case when p_promo_sk is not null then 1 else 0 end) promo
      ,count(*) total_cnt
from catalog_sales
join inventory on (cs_item_sk = inv_item_sk)
join warehouse on (w_warehouse_sk=inv_warehouse_sk)
join item on (i_item_sk = cs_item_sk)
join customer_demographics on (cs_bill_cdemo_sk = cd_demo_sk)
join household_demographics on (cs_bill_hdemo_sk = hd_demo_sk)
join date_dim date_dim1 on (cs_sold_date_sk = date_dim1.d_date_sk)
join date_dim date_dim2 on (inv_date_sk = date_dim2.d_date_sk)
join date_dim date_dim3 on (cs_ship_date_sk = date_dim3.d_date_sk)
left outer join promotion on (cs_promo_sk=p_promo_sk)
left outer join catalog_returns on (cr_item_sk = cs_item_sk and cr_order_number = cs_order_number)
where date_dim1.d_week_seq = date_dim2.d_week_seq
  and inv_quantity_on_hand < cs_quantity
  and date_dim3.d_date > DATE_ADD(date_dim1.d_date, interval 3 day)
  and hd_buy_potential = '>10000'
  and date_dim1.d_year = 1998
  and cd_marital_status = 'D'
  and cd_dep_count between 9 and 11
  and i_category IN ('Books', 'Children', 'Home')
  and cs_wholesale_cost BETWEEN 43 AND 63
group by i_item_desc,w_warehouse_name,date_dim1.d_week_seq
order by total_cnt desc, i_item_desc, w_warehouse_name, d_week_seq
limit 100;


