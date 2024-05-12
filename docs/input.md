---
title: ELLA's Inputs
layout: default
nav_order: 4
---

ELLA takes two pandas data frames as inputs. 

1. Gene expression and cell center 

   A pandas data frame (`expr`) with a few columns:

   - spatial gene expression including the coordinates (`x`, `y`) and the corresponding counts (`umi`) 
   - cell center (`centerX`, `centerY`)
   - and, cell type (`type`), cell ID (`cell`), gene ID (`gene`)
	
	Here's how the data frame looks like:
	<div style="margin: 0 auto; text-align: center;"> 

  	<img src="{{ site.baseurl }}/images/input_expr_df.png" width="500" />
  	</div>	
  	
  	And here's how the expression of one gene (marked as dots in red) and the cell center (marked as "cross" in red) look like:
  	<div style="margin: 0 auto; text-align: center;"> 
  	<img src="{{ site.baseurl }}/images/input_expr.png" width="500" />
  	</div>	

2. Cell segmentation
	A pands dataframe (`cell_seg`) with 3 columns:
	- cell ID (`cell`)
	- the coordinates of points that characterize the cell segmentation boundary or the coordinates of points that characterize the cell segmentation mask (`cell_seg`).
	
	Here's how the data frame looks like:
	<div style="margin: 0 auto; text-align: center;"> 

  	<img src="{{ site.baseurl }}/images/input_cellseg_df.png" width="125" />
  	</div>	
  	
  	And here's how it actually looks like (in red solid line):
  	<div style="margin: 0 auto; text-align: center;"> 
  	<img src="{{ site.baseurl }}/images/input_cellseg.png" width="500" />
  	</div>	

3. [Optional, for visualization purpose ONLY] Nucleus segmentation 

	A pands dataframe (`nucleus_seg`) with 3 columns:
	- cell ID (`cell`)
	- the coordinates of points that characterize the nucleus segmentation boundary or the coordinates of points that characterize the nucleus segmentation mask (`nucleus_seg`).
	
	Here's how the data frame looks like:
	<div style="margin: 0 auto; text-align: center;"> 

  	<img src="{{ site.baseurl }}/images/input_nucleusseg_df.png" width="125" />
  	</div>	
  	
  	And here's how it actually looks like (in darkgray dashed line):
  	<div style="margin: 0 auto; text-align: center;"> 
  	<img src="{{ site.baseurl }}/images/input_nucleusseg.png" width="500" />
  	</div>	





