# Capstone2
**Denver the City that I love...to own a rental property in!**

The objective is to evaluate the possibility of predicting how much profit an Airbnb property will produce given several features, primarily its location.  Essentially, if someone would like to purchase a house to rent out on Airbnb what areas are likely to have low vacancy coupled with,  what is the maximum price they're able to charge in that area.  To investigate the features most likely to be predictors of price and occupancy, many comparisons of the features and subgroups of the features are taken into consideration.

First, the a heat map of the density of listings is created.  The darker color indicates higher density:

![density_heat_map](https://github.com/cody1212/Capstone2/blob/master/images/density_heat_map.png)

Secondly, a heat map of price across all listings.  Again the darker (or more red) the higher the price:

![price_heat_map](https://github.com/cody1212/Capstone2/blob/master/images/price_heat_map.png)

Third, a heat map of availability.  Here the lighter or more green the color indicates greater occupancy  and the darker the color or more red, indicates higher availability.

![availability_heat_map](https://github.com/cody1212/Capstone2/blob/master/images/availability_heat_map.png)

Looking at these heat maps one can grasp a general concept of profitability by location simply by identifying areas where the price heat map is dark and the availability heat map is light indicating the property is being rented regularly (light or green on availability map) and for a high price (dark or red on price heat map).  Someone with a very good understanding of the city can already begin to make some inferences with this information, like what the profitability looks like for properties close to a light rail station, shopping, restaurants, event centers, DTC, downtown, near the airport etc.

An in depth look requires some features being split into sub groups to more accurately evaluate their relation to other features.  This example shows (like all pricing subgroups) a majority of properties being rented for over $200 have very high availability, and therefore, low occupancy over the next 90 day period.  While still many properties in the lower price ranges (of this >$200 subgroup) have a high occupancy as well.  Most interesting is the surprisingly high number of properties in the $1000 price range having significant occupancy numbers over the next 90 days, in fact, more occupancy than not!

![joint_plots_by_price_bin](https://github.com/cody1212/Capstone2/blob/master/images/plots_by_price_bin.png)