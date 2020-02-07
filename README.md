# Capstone2
**Denver the City that I love...to own a rental property in!**

The objective is to evaluate the possibility of predicting how much profit an Airbnb property will produce given several features, primarily its location.  Essentially, if someone would like to purchase a house to rent out on Airbnb what areas are likely to have low vacancy coupled with,  what is the maximum price they're able to charge in that area.  To investigate the features most likely to be predictors of price and occupancy, many comparisons of the features and subgroups of the features are taken into consideration.

First, the a heat map of the density of listings is created.  The darker color indicates higher density:

![density_heat_map](https://github.com/cody1212/Capstone2/blob/master/images/density_heat_map.png)

Secondly, a heat map of price across all listings.  Again the darker (or more red) the higher the price:

![price_heat_map](https://github.com/cody1212/Capstone2/blob/master/images/price_heat_map.png)

Third, a heat map of availability.  Here the lighter or more green the color indicates greater occupancy  and the darker the color or more red, indicates higher availability.

![availability_heat_map](https://github.com/cody1212/Capstone2/blob/master/images/availability_heat_map.png)

Looking at these heat maps one can grasp a general concept of profitability by location simply by identifying areas where the price heat map is dark and the availability heat map is light indicating the property is being rented regularly (light or green on availability map) and for a high price (dark or red on price heat map).  Someone with a very good understanding of the city can already make some inferences with this information, like what the profitability looks like for properties close to a light rail station, shopping, restaurants, event centers, DTC, downtown, near the airport etc.

An in depth look requires some features being split into sub groups to more accurately evaluate their relation to other features.  This example shows (like all pricing subgroups) a majority of properties being rented for over $200 have very high availability, and therefore, low occupancy over the next 90 day period.  While still many properties in the lower price ranges (of this >$200 subgroup) have a high occupancy.  Most interesting is the surprisingly high number of properties in the $1000 price range having significant occupancy numbers over the next 90 days, in fact, more occupancy than not!

![joint_plots_by_price_bin](https://github.com/cody1212/Capstone2/blob/master/images/joint_plots_and_count_histograms.png)

This table shows the type of housing counts grouped by area and whether it's an MDU or SDU.

![area_housing_type_table](https://github.com/cody1212/Capstone2/blob/master/images/housing_type_table.png)

Another noticable item in the above histograms is the spikes in pricing at each $50 mark indicating many listings could be charging more since there is less than a one day difference in the mean availability of houses on $50 marks, in fact listings at the $50 dollar marks are the (marginally) higher of the 2.

Next the prices and occupancy are looked at from a more zoomed in level, taking into account first the area then the zipcode.

![area_box_plots_by_price](https://github.com/cody1212/Capstone2/blob/master/images/price_by_area_box_plot.png)
![area_box_plots_by_occupancy](https://github.com/cody1212/Capstone2/blob/master/images/reservations_by_area_box_plot.png)
![zip_price](https://github.com/cody1212/Capstone2/blob/master/images/price_zips.png)
![zip_occupancy](https://github.com/cody1212/Capstone2/blob/master/images/occupancy_zips.png)
![zip_cleaning_fee](https://github.com/cody1212/Capstone2/blob/master/images/zipcode_cleaning.png)

Finally, a Random Forest Regression is conducted to investigate how well a machine learning model can predict housing prices when trained on this data, as well as, provide some insight into what features can descibe the largerst percentage of the variance in the prices the model predicts and the actual prices of the listings.  This shows the unanimous greatest feature importance score goes to cleaning fee, consistently followed by number of bathrooms, with a fluctuating mix of bedrooms, 365_day availability, location, security deposit and housing type.  Since, RFs are not deterministic, the feature importances do change each time the model is run indicated by the varied tables below.  The most surprising maybe of those results is the type of housing was not given a greater feature importance.  

![feature_importance1](https://github.com/cody1212/Capstone2/blob/master/images/feature_imps1.png)
![feature_importance2](https://github.com/cody1212/Capstone2/blob/master/images/feature_imps2.png)
![feature_importance3](https://github.com/cody1212/Capstone2/blob/master/images/feature_imps3.png)

Another couple other Random Forest Regression analysis was done by removing the 30, 60, and 90 availability, focusing on the 365 availability as the target.  And an additional one with cleaning fee as the target.  These were the results.

![rmse_score](https://github.com/cody1212/Capstone2/blob/master/images/rmse_availability_365.png)
![feature_importance4](https://github.com/cody1212/Capstone2/blob/master/images/feature_imps_4.png)

![rmse_score2](https://github.com/cody1212/Capstone2/blob/master/images/cleaning_fee_rf_score.png)
![feature_importance5](https://github.com/cody1212/Capstone2/blob/master/images/feat_imps_avail.png)

![rf](https://github.com/cody1212/Capstone2/blob/master/images/RF.png)
![tent_guy](https://github.com/cody1212/Capstone2/blob/master/images/tent_guy.png)


