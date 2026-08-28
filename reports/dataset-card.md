# Dataset card — Finland GeoAI v1

## Summary

The dataset pairs real Copernicus Sentinel-2 Level-2A surface-reflectance imagery with ESA WorldCover 2021 v200 categorical land cover. The acquisition window and labels are both from 2021 to reduce temporal mismatch. It contains four bounded Finnish AOIs, 850 non-overlapping 64×64 patches and 3,481,059 valid labelled pixels. Generated rasters are ignored; the manifest, configuration and acquisition code are source-controlled.

## Dataset decision

The National Land Survey of Finland route was audited first. NLS orthophotos and the Topographic Database are open data, but reproducible WCS and OGC API Processes access requires a personal API key. NLS documents that requirement for both the [orthophoto WCS](https://www.maanmittauslaitos.fi/ortokuvien-ja-korkeusmallien-kyselypalvelu/tekninen-kuvaus) and [download-related services](https://www.maanmittauslaitos.fi/en/e-services/mapsite/related-services). Under the project's credential stop rule, implementation therefore moved directly to the zero-credential Sentinel-2 + WorldCover route. There is no abandoned NLS downloader.

## Sources and licensing

| Role | Source | Version / collection | Access | Licence |
| --- | --- | --- | --- | --- |
| Image | Copernicus Sentinel-2 Level-2A | Earth Search `sentinel-2-l2a` | HTTPS range reads from public AWS COGs | Copernicus data; attribution and terms apply |
| Label | ESA WorldCover | 2021 v200, 10 m | Public AWS COGs | CC BY 4.0 |

WorldCover documents public S3 access, its 2021 v200 version and licence on the official [data access page](https://esa-worldcover.org/en/data-access). Required attribution: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium.

## AOIs, scenes and split

| AOI | Role | Sentinel-2 item | Date | Scene cloud metadata | Patches |
| --- | --- | --- | --- | ---: | ---: |
| Helsinki | train | `S2A_35VLG_20210703_2_L2A` | 2021-07-03 | 0.0107% | part of 269 train |
| Lahti | train | `S2B_35VMH_20210618_1_L2A` | 2021-06-18 | 0.0071% | part of 269 train |
| Tampere | validation | `S2A_35VLJ_20210716_1_L2A` | 2021-07-16 | 0.0003% | 266 |
| Oulu | protected test | `S2A_35WMN_20210702_1_L2A` | 2021-07-02 | 0.0001% | 315 |

Exact WGS84 boxes, target bounds, URLs and patch records are in `artifacts/dataset-manifest-v1.json`. Scene-level cloud cover is only a search filter; SCL codes 0, 1, 3, 8, 9, 10 and 11 are masked per pixel for no-data, saturation, shadow, cloud/cirrus and snow.

## Spatial processing

- Source Sentinel scenes: EPSG:32635; source resolution is 10 m for B02/B03/B04/B08 and 20 m for SCL.
- WorldCover source: EPSG:4326, 10 m nominal product.
- Target: EPSG:3067 (ETRS-TM35FIN), 10 m square grid aligned to metric multiples of 10.
- Continuous reflectance reprojection: bilinear.
- Categorical SCL and WorldCover reprojection: nearest-neighbour only.
- Patches: 64×64, stride 64; no overlap inside an AOI.
- Acceptance: at least 90% valid labelled pixels.
- Reflectance normalization: divide by 10,000, then train-only mean `[0.04546451, 0.05896357, 0.04728721, 0.21076994]` and standard deviation `[0.03435183, 0.03790009, 0.04505760, 0.14147774]`.

## Class mapping and support

The mapping was declared after inspecting Finnish support. Rare source classes are merged rather than silently removed.

| Target | WorldCover source codes | Train | Validation | Test | Reasoning |
| --- | --- | ---: | ---: | ---: | --- |
| forest | 10 | 51.15% | 41.10% | 52.27% | supported primary boreal class |
| shrub/grass | 20, 30 | 3.22% | 1.34% | 7.87% | individually sparse, spectrally related open vegetation |
| cropland | 40 | 5.01% | 1.40% | 0.58% | kept separate despite geographic imbalance |
| built-up | 50 | 15.07% | 21.76% | 19.27% | supported urban class |
| other natural | 60, 70, 90, 95, 100 | 0.66% | 0.77% | 2.01% | rare bare/snow/wetland/mangrove/moss codes retained as one disclosed group |
| water | 80 | 24.88% | 33.64% | 18.00% | strongly supported Finnish class |

WorldCover code 0 and invalid SCL pixels map to ignore index 255. They never enter loss or metrics. No labelled source class is silently discarded.

![Class proportions by held-out geography](figures/class-distribution.png?v=20260828-geospatial-ai-v1)

## Leakage controls

The split unit is the AOI, not the patch. An AOI has exactly one role, tiles do not cross roles, patches have no overlap, and pairwise train/validation/test bounding-box intersections are a blocking validation error. Patch IDs and content hashes must be unique. The committed split is `held-out-aoi/v1`; changing the protected test requires a new version and documented data defect.

## Known limitations

- Four small AOIs are not statistically representative of Finland, seasons or acquisition conditions.
- WorldCover is itself a model-derived product, not hand-labelled ground truth; boundaries and rare classes contain label noise.
- Even within 2021, Sentinel acquisition date and the WorldCover annual product are not perfectly simultaneous.
- Each split is tied to one or two scenes; geography, land-cover mix and scene characteristics are confounded.
- Snow is masked, so winter performance is unknown.
- The six-class merge reduces thematic detail. The rare `other_natural` group is heterogeneous and proved unlearnable at this scale.
- Non-overlap removes direct patch leakage but does not create independent repeated measurements of Finland-wide generalization.
