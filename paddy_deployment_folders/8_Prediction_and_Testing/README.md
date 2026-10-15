# Prediction and Testing

Tests the deployed model (files from `7_Model_Deployment`) with different inputs. Run: `python test_predictions.py`

## A. Scenario tests
Baseline farm = dataset median values (Cuddalore, CO_43, alluvial soil, dry nursery, 4 ha). Each test changes only the listed inputs.

| Test | Description | Predicted (kg/ha) | Total (kg) | Result |
|---|---|---|---|---|
| T1 | Typical farm (all defaults) | 6127.4 | 24509.6 | PASS |
| T2 | High fertilizer (max Urea and Potash) | 6163.4 | 24653.7 | PASS |
| T3 | Low fertilizer (min Urea and Potash) | 6000.5 | 24002.1 | PASS |
| T4 | Variety = ponmani | 6098.7 | 24394.9 | PASS |
| T5 | Variety = delux ponni | 6135.8 | 24543.4 | PASS |
| T6 | Soil = clay | 6132.9 | 24531.7 | PASS |
| T7 | Nursery = wet | 6134.6 | 24538.3 | PASS |
| T8 | Agriblock = Panruti | 6127.4 | 24509.6 | PASS |
| T9 | Small farm (min hectares) | 6075.5 | 6075.5 | PASS |
| T10 | Large farm (max hectares) | 6135.9 | 36815.4 | PASS |
| T11 | All inputs at minimum | 5789.9 | 5789.9 | PASS |
| T12 | All inputs at maximum | 5960.1 | 35760.5 | PASS |
| T13 | Edge case: unseen Variety value | 6130.9 | 24523.7 | PASS |

PASS = prediction is a valid number within the realistic paddy yield range (2,000 to 10,000 kg/ha).

**Observations**
- All 13 tests produced valid predictions, including the unseen-category edge case (the app does not crash).
- Predicted yield per hectare stays close to 6,000 kg/ha across inputs. Raising Urea and Potash from minimum to maximum changes the prediction by about 160 kg/ha (T3 vs T2), and the all-minimum vs all-maximum inputs differ by about 170 kg/ha (T11 vs T12).
- Total yield scales with farm size (T9 vs T10), as expected.
- Changing Agriblock gave the same prediction as the baseline, so this feature has almost no influence on the model.
- Because the dataset has limited variation (R2 about 0.50), the app should be used as an estimate, not an exact forecast.

## B. Batch test on 25 real records
`batch_test_results.csv` compares actual and predicted yield for 25 records: mean absolute error is about 145 kg/ha (2.4 %).
These records were part of the final training data, so this only confirms that the whole pipeline works end to end.
Accuracy on unseen data is in `3_Performance_Evaluation_Metrics` (MAE about 155 kg/ha, R2 about 0.50).

## C. Testing the live app (manual)
1. Open the live link in `9_Deployment_Link`.
2. Keep the default values and click **Predict yield**. Expected: about 6,127 kg/ha and about 24,510 kg total for 4 ha.
3. Change Urea / Potash / Hectares and predict again. The result and total should update.
4. Add screenshots of these tests in this folder (for example `screenshot_default.png`, `screenshot_changed_inputs.png`).

Output files: `scenario_test_results.csv`, `batch_test_results.csv`
