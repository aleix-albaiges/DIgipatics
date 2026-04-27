------------
| Contents |
------------

Version 1: 2020-05


---------------
| Directories |
---------------
D1. images
	Prostate histology patches of slides at 10X magnification. Patches are extracted with size 512 pixels and overlap of 50% among them. Only the patches with more than 20% of tissue are selected.
	The first prefix in filename indicated the slide ID (e.g. 16B0001851 in '16B0001851_Block_Region_1_0_0_xini_6803_yini_59786').
	The patch position in the slide is indicated by the prefixes: Block_Region_region_y_x. Each slide have different regions, and the patch position is indicated in _y_x_.
D2. masks
	Mask with encoded anotations of the images in D1.
		0 --> non cancerous
		1 --> GG3	
		2 --> GG4
		3 --> GG5
D3. partition
	Tables with the proposed patient-based cross-validation partition of the database, Gleason grades labels, and ground truth of cribriform patterns.
	The non-cancerous patches in this partition are obtained from slides classified as non cancerous.
	The labels in cancerous patches are obtained by majority voting of the annotations (masks).

----------
| Tables |
----------
T1. wsi_labels
	Table with the wsi ID, patient ID and its respective Gleason score (primary and secondary grades).
