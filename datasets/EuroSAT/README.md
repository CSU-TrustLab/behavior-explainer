# EuroSAT

**EuroSAT** is a land use and land cover classification dataset based on Sentinel-2 satellite imagery from the European Space Agency (ESA). It covers 13 spectral bands and consists of 27,000 labeled geo-referenced patches across 10 classes.

## Structure

```
EuroSAT/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

## Classes (10)

| Class | Description |
|---|---|
| AnnualCrop | Annually harvested crops |
| Forest | Forest areas |
| HerbaceousVegetation | Grasslands and herbaceous cover |
| Highway | Roads and highways |
| Industrial | Industrial facilities |
| Pasture | Permanent grassland / pasture |
| PermanentCrop | Vineyards, orchards, etc. |
| Residential | Residential urban areas |
| River | Rivers and canals |
| SeaLake | Sea and lakes |

## Statistics

- 27,000 images total (2,000–3,000 per class)
- Image size: 64×64 pixels (RGB version)

## Download

> **The dataset images are not tracked in this repository.**
> Please download the RGB version of EuroSAT from the official repository:
>
> <https://github.com/phelber/EuroSAT>
>
> Direct ZIP: <https://madm.dfki.de/files/sentinel/EuroSAT.zip>

After downloading, place the class folders under `datasets/EuroSAT/` following the structure above.

## Citation

```bibtex
@article{helber2019eurosat,
  title   = {EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author  = {Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume  = {12},
  number  = {7},
  pages   = {2217--2226},
  year    = {2019},
  doi     = {10.1109/JSTARS.2019.2918242}
}
```
