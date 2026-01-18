# Sample Data

This directory contains sample SAP-like data for testing.

## Files

- `sample_materials.csv` - Material master data
- `sample_suppliers.csv` - Supplier master data

## Generation

To regenerate sample data:
```python
from src.sap_connector import create_sample_sap_data

df = create_sample_sap_data(n_materials=1000)
df.to_csv('data/sample_materials.csv', index=False)
```

## Schema

### sample_materials.csv

| Column | Description | Example |
|--------|-------------|---------|
| MATNR | Material Number | MAT000001 |
| MAKTX | Material Description | Steel Bolt M8x50 DIN 933 |
| MATKL | Material Group | BOLTS |
| MTART | Material Type | FERT |
| MEINS | Base Unit of Measure | PC |