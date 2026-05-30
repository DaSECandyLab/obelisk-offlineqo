SELECT w_state ,
         i_item_id ,
         sum(
    CASE
    WHEN ( d_date < '2002-06-01') THEN
    cs_sales_price - coalesce(cr_refunded_cash,0)
    ELSE 0 END) AS sales_before , sum(
    CASE
    WHEN ( d_date >= '2002-06-01') THEN
    cs_sales_price - coalesce(cr_refunded_cash,0)
    ELSE 0 END) AS sales_after
FROM tpcds50.catalog_sales LEFT OUTER
JOIN tpcds50.catalog_returns
    ON ( cs_order_number = cr_order_number
        AND cs_item_sk = cr_item_sk) , warehouse , item , date_dim
WHERE i_current_price
    BETWEEN 0.99
        AND 1.49
        AND i_item_sk = cs_item_sk
        AND cs_warehouse_sk = w_warehouse_sk
        AND cs_sold_date_sk = d_date_sk
        AND d_date
    BETWEEN '2002-05-01'
        AND '2002-07-01'
GROUP BY  w_state, i_item_id
ORDER BY  w_state, i_item_id LIMIT 100 