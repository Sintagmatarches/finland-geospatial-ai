# NLS source audit

Audit date: 2026-09-01.

## Orthophotos

The selected product is the National Land Survey of Finland digital colour orthophoto (`ortokuva`),
distributed as source JPEG2000 map sheets. NLS documents a typical 0.5 m pixel size and
ETRS-TM35FIN. Public MapSite coverage metadata confirms 2025 photography for `L3243F`, `L3243D`,
`L3243H`, `L3244C`, and `L3244H`.

The reproducible download route is the official NLS OGC API Processes service using process
`ortoilmakuva_karttalehti`, `dataSetInput=ortokuva`, `fileFormatInput=JPEG2000`, and `yearInput=2025`.
The service requires a personal NLS API key. The client uses HTTP Basic authentication, keeps the key
outside files, follows asynchronous job links, downloads source results, and writes SHA-256 receipts.

## Topographic Database

The selected label source is the NLS Topographic Database GeoPackage, bounded to
`[194000, 6744000, 212000, 6762000]` in EPSG:3067. The official process is
`maastotietokanta_bbox` with `themeInput=maastotietokanta_kaikki`.

Before freezing the class map, the official public L324 GeoPackage test product with snapshot date
2026-04-01 was inspected. It contains the declared layers for water, cultivated land, precise built
surfaces, and open-natural surfaces. It does not provide a complete dense-forest polygon class;
therefore the residual class is named `other_land`, not forest.

The AOI was selected using measured support, not convenience. In the audited snapshot, training sheet
`L3244C` contains 5.815 km² mapped water and 1.617 km² precise built surfaces, while `L3244H`
contains 15.458 km² cultivated land. Validation sheet `L3243D` contains 9.297 km² water. The two
sealed-test sheets differ materially: `L3243F` contains 8.786 km² cultivated land and 5.067 km²
open-natural surfaces; `L3243H` contains 6.078 km² cultivated and 7.258 km² open-natural surfaces.
The complete feature counts, areas, layer inventory, and source SHA-256 are stored in
`artifacts/source-audit.json`.

## Temporal decision

The 2025 imagery and the current Topographic Database snapshot are close but not simultaneous.
Acquisition year and vector snapshot date are recorded separately. Transition pixels receive a
1.5 m uncertainty mask, and obvious source omissions remain documented rather than silently relabelled.

## Official references

- [Orthophotos product description](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/orthophotos)
- [Topographic Database product description](https://www.maanmittauslaitos.fi/en/maps-and-spatial-data/datasets-and-interfaces/product-descriptions/topographic-database)
- [OGC API Processes service](https://www.maanmittauslaitos.fi/paikkatiedon-tiedostopalvelu)
- [OGC API Processes technical examples](https://www.maanmittauslaitos.fi/paikkatiedon-tiedostopalvelu/tekninen-kuvaus)
- [NLS API-key instructions](https://www.maanmittauslaitos.fi/en/rajapinnat/api-avaimen-ohje)
