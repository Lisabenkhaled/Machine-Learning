# Description des features

Ce rapport résume les variables explicatives (numériques) les plus dispersées.

## Statistiques descriptives

| feature | count | mean | std | min | 25% | 50% | 75% | max | missing_ratio |
|---|---|---|---|---|---|---|---|---|---|
| load_forecast_lag_24 | 7197.0000 | 52277.5765 | 11415.7988 | 30515.0000 | 43701.0000 | 50104.0000 | 60454.0000 | 85145.0000 | 0.1517 |
| load_forecast_lag_1 | 7220.0000 | 52276.9476 | 11399.1588 | 30515.0000 | 43740.2500 | 50120.0000 | 60425.0000 | 85145.0000 | 0.1490 |
| load_forecast | 7221.0000 | 52277.4541 | 11398.4506 | 30515.0000 | 43741.0000 | 50121.0000 | 60425.0000 | 85145.0000 | 0.1489 |
| nuclear_power_available | 8483.0000 | 33695.3546 | 7414.0691 | 22285.0000 | 28620.0000 | 30710.0000 | 38897.0000 | 50122.0000 | 0.0001 |
| wind_power_forecasts_average | 8460.0000 | 4065.7286 | 3019.7345 | 592.0000 | 1898.7500 | 3017.0000 | 5266.0000 | 14629.0000 | 0.0028 |
| wind_power_forecasts_average_lag_1 | 8459.0000 | 4064.4798 | 3017.7277 | 592.0000 | 1898.5000 | 3017.0000 | 5264.0000 | 14613.0000 | 0.0029 |
| wind_power_forecasts_average_lag_24 | 8436.0000 | 4050.7050 | 3006.6327 | 592.0000 | 1895.5000 | 3008.0000 | 5235.0000 | 14613.0000 | 0.0057 |
| solar_power_forecasts_average_lag_24 | 8460.0000 | 2010.3417 | 2774.7988 | 0.0000 | 0.0000 | 212.0000 | 3730.5000 | 10127.0000 | 0.0028 |
| solar_power_forecasts_average_lag_1 | 8483.0000 | 2005.8884 | 2772.5392 | 0.0000 | 0.0000 | 209.0000 | 3714.5000 | 10127.0000 | 0.0001 |
| solar_power_forecasts_average | 8484.0000 | 2005.8735 | 2772.3761 | 0.0000 | 0.0000 | 209.0000 | 3713.7500 | 10127.0000 | 0.0000 |
| coal_power_available | 8483.0000 | 2712.8325 | 475.5010 | 2226.0000 | 2226.0000 | 2806.0000 | 3386.0000 | 3386.0000 | 0.0001 |
| gas_power_available | 8483.0000 | 11349.9248 | 474.9068 | 9769.0000 | 11060.0000 | 11480.0000 | 11915.0000 | 11963.0000 | 0.0001 |
| gas_power_available_lag_1 | 8482.0000 | 11349.8546 | 474.8908 | 9769.0000 | 11060.0000 | 11480.0000 | 11915.0000 | 11963.0000 | 0.0002 |
| gas_power_available_lag_24 | 8459.0000 | 11348.2364 | 474.5195 | 9769.0000 | 11060.0000 | 11480.0000 | 11915.0000 | 11963.0000 | 0.0029 |
| wind_power_forecasts_std_lag_24 | 8436.0000 | 119.7005 | 125.3661 | 1.8302 | 44.9790 | 82.6208 | 148.0636 | 1667.6881 | 0.0057 |
| wind_power_forecasts_std_lag_1 | 8459.0000 | 119.7949 | 125.2653 | 1.8302 | 45.0268 | 82.7236 | 148.4671 | 1667.6881 | 0.0029 |
| wind_power_forecasts_std | 8460.0000 | 119.7881 | 125.2594 | 1.8302 | 45.0273 | 82.7164 | 148.4668 | 1667.6881 | 0.0028 |
| solar_power_forecasts_std_lag_24 | 8460.0000 | 25.3994 | 43.5049 | 0.0000 | 0.0000 | 4.1456 | 36.2009 | 745.2613 | 0.0028 |
| solar_power_forecasts_std_lag_1 | 8483.0000 | 25.3587 | 43.4654 | 0.0000 | 0.0000 | 4.1275 | 36.1706 | 745.2613 | 0.0001 |
| solar_power_forecasts_std | 8484.0000 | 25.3595 | 43.4629 | 0.0000 | 0.0000 | 4.1283 | 36.1672 | 745.2613 | 0.0000 |
| spot_id_delta | 8484.0000 | 0.2836 | 41.7484 | -1567.3535 | -14.4056 | -0.7738 | 12.7591 | 658.9613 | 0.0000 |
| hour | 8484.0000 | 11.5052 | 6.9177 | 0.0000 | 6.0000 | 12.0000 | 17.2500 | 23.0000 | 0.0000 |
| month | 8484.0000 | 6.5332 | 3.4519 | 1.0000 | 4.0000 | 7.0000 | 10.0000 | 12.0000 | 0.0000 |
| dayofweek | 8484.0000 | 2.9840 | 2.0073 | 0.0000 | 1.0000 | 3.0000 | 5.0000 | 6.0000 | 0.0000 |
| hour_sin | 8484.0000 | -0.0003 | 0.7074 | -1.0000 | -0.7071 | 0.0000 | 0.7071 | 1.0000 | 0.0000 |

## Lecture métier (interprétation rapide)

- `load_forecast` : proxy direct de la demande; une demande élevée pousse souvent les prix à la hausse.
- `gas_power_available` : le gaz est fréquemment marginal sur le marché, donc fortement lié au prix spot.
- `wind/solar *_average` : plus la production ENR anticipée est forte, plus la pression baissière sur les prix est probable.
- `*_std` : mesure l'incertitude des prévisions ENR; l'incertitude peut augmenter la volatilité intraday.
- Variables calendaires (`hour`, `dayofweek`, sin/cos) : capturent les cycles journaliers et hebdomadaires.
