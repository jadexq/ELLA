---
title: ELLA's Inputs
layout: default
nav_order: 5
---

## Inputs

ELLA mainly takes two pandas data frames as inputs. 

**1. Gene expression with nuclear center** <br>
A pandas data frame (`expr`) with a few columns:
- spatial gene expression including the coordinates (`x`, `y`) and the corresponding counts (`umi`) 
- cell center (`centerX`, `centerY`)
- cell type (`type`), cell ID (`cell`), gene ID (`gene`)  <br>
- and the total number expression counts of cells (`sc_total`), rows corresponding to the same cell should have the same value for this column.
  

Here's how the data frame looks like:
<div style="margin: 0 auto; text-align: center;"> 
  <img src="{{ site.baseurl }}/images/input_expr_df.png" width="500" />
</div>

And here's how the expression of one gene (dots in red) and the cell center (crosses  in red) looks like:
<div style="margin: 0 auto; text-align: center;"> 
  <img src="{{ site.baseurl }}/images/input_expr.png" width="500" />
</div>


<br>
**2. Cell segmentation** <br>
A pands data frame (`cell_seg`) with 3 columns:
- cell ID (`cell`)
- the coordinates of points that characterize the cell segmentation boundary (`cell_seg`). <br>

Here's how the data frame looks like:
<div style="margin: 0 auto; text-align: center;"> 
  <img src="{{ site.baseurl }}/images/input_cellseg_df.png" width="125" />
</div>

And here's how it actually looks like (in red solid line):
<div style="margin: 0 auto; text-align: center;"> 
  <img src="{{ site.baseurl }}/images/input_cellseg.png" width="500" />
</div>	


<br>
**Nucleus segmentation** <br>
[Optional, for visualization purpose ONLY] <br>

A pands dataframe (`nucleus_seg`) with 3 columns:
- cell ID (`cell`)
- the coordinates of points that characterize the nucleus segmentation boundary (`nucleus_seg`). <br>

Here's how the data frame looks like:
<div style="margin: 0 auto; text-align: center;"> 
  <img src="{{ site.baseurl }}/images/input_nucleusseg_df.png" width="125" />
</div>	

And here's how it actually looks like (in darkgray dashed line):
<div style="margin: 0 auto; text-align: center;"> 
  <img src="{{ site.baseurl }}/images/input_nucleusseg.png" width="500" />
</div>	


<br>
**Other required inputs**<br>
- `types` a list corresponding to all cell types.
- `cells` a dictionary of lists corresponding to list of cells in each cell type.
- `cells_all` a list of all cells across cell types.
- `genes` a dictionary of lists corresponding to list of genes in each cell type.

How about tweak your own data into the format that ELLA takes and have a try!



