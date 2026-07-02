# Supported Data Formats

InversionAI supports multiple geophysical data formats. The system auto-detects formats on import and can convert between them.

## CSV (Comma-Separated Values)

The simplest and most portable format. Recommended for new users.

### Gravity CSV

```csv
x,y,z,gobs,std
0.0,0.0,0.0,-5.23,0.05
100.0,0.0,0.0,-4.87,0.05
200.0,0.0,0.0,-3.14,0.05
```

### Magnetic CSV

```csv
x,y,z,tmi,std
0.0,0.0,80.0,152.3,1.0
100.0,0.0,80.0,98.7,1.0
200.0,0.0,80.0,-45.2,1.0
```

### Column Definitions

| Column | Unit | Description |
|--------|------|-------------|
| `x` | metres | Easting coordinate |
| `y` | metres | Northing coordinate |
| `z` | metres | Elevation (positive up) |
| `gobs` | mGal | Observed gravity anomaly |
| `tmi` | nT | Total magnetic intensity anomaly |
| `std` | same as data | Standard deviation (uncertainty) |

### Coordinate System

- Right-handed: X=East, Y=North, Z=Up
- Local or projected coordinates (not geographic lat/lon)
- All stations should use the same datum and projection

---

## UBC Mesh Format

Used by UBC-GIF codes (GRAV3D, MAG3D). Supported for compatibility with existing workflows.

### Mesh File (`.msh`)

```
NE NN NZ
E0 N0 Z0
dE1 dE2 dE3 ... dE_NE
dN1 dN2 dN3 ... dN_NN
dZ1 dZ2 dZ3 ... dZ_NZ
```

**Example:**

```
10 10 5
0.0 0.0 0.0
100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0
100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0 100.0
50.0 50.0 50.0 50.0 50.0
```

- Line 1: Number of cells in East, North, Down directions
- Line 2: Origin (South-West-Top corner)
- Lines 3+: Cell widths in each direction

### Observation File (`.obs` / `.grv`)

```
N_observations
x1 y1 z1 data1 std1
x2 y2 z2 data2 std2
...
```

### Model File (`.den` / `.sus`)

One value per line, ordered by: East (fastest), North, Down (slowest).

```
0.001
-0.002
0.015
...
```

---

## Geosoft XYZ

Exported from Oasis montaj. Comment lines start with `/`.

```
/ Geosoft XYZ export
/ Survey: Regional Gravity 2024
/ Line  Easting  Northing  Elevation  Gravity  StdDev
/
   0.0    0.0    0.0   -5.23   0.05
 100.0    0.0    0.0   -4.87   0.05
 200.0    0.0    0.0   -3.14   0.05
```

### Characteristics

- Comment lines start with `/`
- Data is space-separated (fixed-width or free-format)
- Column order defined in comments (no standard)
- Line breaks indicated by blank lines or `/` markers

InversionAI reads the first non-comment line to determine column count, then maps columns by position (x, y, z, data, std).

---

## Tomofast Native Format

Tomofast-X uses plain text space-separated files with no headers.

### Data File

```
x1 y1 z1 data1 std1
x2 y2 z2 data2 std2
...
```

No header line. Columns are space-separated. Order: x, y, z, observed value, uncertainty.

### Mesh File

Tomofast-X uses its own mesh description in the parameter file (Parfile). The mesh is defined by:

```
# In Parfile:
nelements_x = 10
nelements_y = 10
nelements_z = 5
origin_x = 0.0
origin_y = 0.0
origin_z = 0.0
cell_size_x = 100.0
cell_size_y = 100.0
cell_size_z = 50.0
```

Tomofast-X supports only regular (uniform cell size) meshes.

---

## Tips and Common Issues

### Unit Conversion

| From | To | Multiply by |
|------|----|------------|
| µGal → mGal | | 0.001 |
| mGal → m/s² | | 1e-5 |
| nT → A/m | | (depends on context) |
| feet → metres | | 0.3048 |
| km → metres | | 1000 |

### Common Issues

**Problem:** Inversion diverges immediately.
**Cause:** Units mismatch. Gravity in µGal when mGal expected, or coordinates in km instead of metres.
**Fix:** Ensure gravity is in mGal, magnetics in nT, coordinates in metres.

**Problem:** Model is all zeros.
**Cause:** Standard deviations too large (over-fitting threshold met immediately).
**Fix:** Check that `std` column is in the same units as the data column.

**Problem:** "File format not recognized" error.
**Cause:** File has BOM characters, mixed line endings, or non-UTF8 encoding.
**Fix:** Save as UTF-8 with Unix line endings. Remove BOM with: `sed -i '1s/^\xEF\xBB\xBF//' file.csv`

**Problem:** Coordinate system mismatch between data and mesh.
**Cause:** Data in geographic (lat/lon) but mesh expects projected coordinates.
**Fix:** Project coordinates to a local system (UTM or custom) before import.

### Coordinate System Notes

- InversionAI uses a **local Cartesian** system internally
- If your data is in lat/lon, convert to UTM or a local projection first
- The Z-axis is positive **upward** for CSV and Geosoft formats
- The Z-axis is positive **downward** for UBC mesh (depth from surface)
- Tomofast-X uses Z positive **downward** internally — InversionAI handles the conversion
